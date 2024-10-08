import json
import logging
import os
import random
import textwrap

from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from gretel_client import Gretel

from .content_validator import ContentValidator
from .data_models import DataField, DataModel
from .evolutionary_strategies import DEFAULT_EVOLUTION_STRATEGIES
from .prompts import (
    CONTENT_CORRECTION_PROMPT,
    FIELD_GENERATION_PROMPT,
    LLM_JUDGE_PROMPT,
    MUTATION_PROMPT,
)
from .text_inference import TextInference


class EvolDataGenerator:
    def __init__(
        self,
        model_definition: DataModel,
        custom_evolutionary_strategies: Optional[List[str]] = None,
    ):
        self.model_definition = model_definition
        self._setup_logging()
        self.gretel = Gretel(api_key=self.model_definition.api_key)
        self.llm = self.gretel.factories.initialize_navigator_api(
            "natural_language", backend_model=self.model_definition.llm_model
        )
        self.custom_evolutionary_strategies = custom_evolutionary_strategies or []
        self.text_inference = TextInference(self.llm, self.logger)
        self.use_reflection = self.model_definition.use_reflection
        self.validators = self._initialize_validators()
        self.evolutionary_strategies = self._initialize_evolution_strategies()

        self.success_count = 0
        self.fail_count = 0
        self.output_filename = None

    def _initialize_evolution_strategies(self) -> List[str]:
        """
        Initialize evolutionary strategies by either using the custom strategies provided
        or falling back to the default strategies.

        Returns:
            List[str]: The list of evolutionary strategies to use.
        """
        if self.custom_evolutionary_strategies:
            self.logger.info("Using custom evolutionary strategies.")
            return self.custom_evolutionary_strategies
        else:
            self.logger.info("Using default evolutionary strategies.")
            return DEFAULT_EVOLUTION_STRATEGIES

    def _select_evolutionary_strategy(self, field: DataField) -> str:
        """
        Select a random evolutionary strategy for a given field. The field's evolutionary strategies
        will default to the global DEFAULT_EVOLUTION_STRATEGIES if no custom strategies are provided.

        Args:
            field (DataFieldDefinition): The field for which to select a strategy.

        Returns:
            str: The selected evolutionary strategy.
        """
        strategies = field.evolution_strategies
        if not strategies:
            self.logger.warning(
                f"No evolution strategies defined for field {field.name}. Skipping evolution."
            )
            return ""
        return random.choice(strategies)

    def _generate_field_value(
        self,
        context: Dict[str, Any],
        field: DataField,
        current_record: Dict[str, Any],
    ) -> Any:
        prompt = FIELD_GENERATION_PROMPT.format(
            generation_instructions=self.model_definition.generation_instructions,
            context=json.dumps(context, indent=2),
            current_record=json.dumps(current_record, indent=2),
            field_name=field.name,
            field_type=field.type,
            field_description=field.description,
        )
        self.logger.debug(f"Prompt for {field.name}:\n{prompt}")

        value = self.text_inference.generate(
            prompt,
            use_reflection=self.use_reflection,
            return_full_reflection=field.store_full_reflection,
            system_message=(
                self.model_definition.generation_instructions
                if not self.use_reflection
                else None
            ),
            temperature=0.8 if self.use_reflection else 0.7,
            max_tokens=2048,
            field_name=field.name,
        )

        parsed_value = self._parse_field_value(value.strip(), field)
        self.logger.debug(f"Generated value for {field.name}: {parsed_value}")
        return parsed_value

    def _mutate_field_value(
        self,
        value: Any,
        evolution_strategy: str,
        field: DataField,
        context: Dict[str, Any],
        current_record: Dict[str, Any],
    ) -> Any:
        prompt = MUTATION_PROMPT.format(
            generation_instructions=self.model_definition.generation_instructions,
            evolution_strategy=evolution_strategy,
            value=value,
            field_name=field.name,
            field_type=field.type,
            field_description=field.description,
            context=json.dumps(context, indent=2),
            current_record=json.dumps(current_record, indent=2),
        )

        return self.text_inference.generate(
            prompt,
            use_reflection=self.use_reflection,
            return_full_reflection=field.store_full_reflection,
            temperature=0.8,
            field_name=field.name,
        )

    def _llm_judge_check(self, generated_record: Dict[str, Any]) -> Tuple[bool, str]:
        self.logger.info("Starting LLM judge check...")

        prompt = LLM_JUDGE_PROMPT.format(
            generated_record=json.dumps(generated_record, indent=2),
            model_definition=self.model_definition.model_dump_json(indent=2),
        )

        self.logger.debug(f"LLM judge prompt:\n{prompt}")

        response = self.text_inference.generate(
            prompt,
            use_reflection=self.use_reflection,
            return_full_reflection=False,
            temperature=0.1,
            field_name="LLM_JUDGE",
        )

        if "FAIL" in response:
            self.logger.info("LLM judge check: FAIL")
            self.logger.debug(f"LLM judge failure reason:\n{response.strip()}")
            return False, response.strip()
        else:
            self.logger.info("LLM judge check: PASS")
            return True, response.strip()

    def _validate_and_correct_field_value(
        self, value: Any, field: DataField
    ) -> Tuple[Any, bool]:
        validator = self.validators.get(field.name)
        if validator:
            error = validator(value)
            if error:
                self.logger.warning(f"Validation error for {field.name}: {error}")
                corrected_value = self._correct_content(value, field.type, error, field)
                if corrected_value != value:
                    if not validator(corrected_value):
                        self.logger.info(f"Repaired value for {field.name}")
                        return corrected_value, True
                    else:
                        self.logger.info(f"Failed to repair value for {field.name}")
                        return corrected_value, False
                else:
                    self.logger.warning(f"Unable to repair value for {field.name}")
                    return value, False
        return value, True

    def _correct_content(
        self,
        content: str,
        content_type: str,
        error_message: str,
        field: DataField,
    ) -> str:

        return self.text_inference.generate(
            CONTENT_CORRECTION_PROMPT.format(
                model_definition=self.model_definition.model_dump_json(indent=2),
                content_type=content_type.upper(),
                error_message=error_message,
                content=content,
            ),
            use_reflection=self.model_definition.use_reflection,
            return_full_reflection=field.store_full_reflection,
            temperature=0.2,
            field_name="Attempting Correction",
        )

    def _generate_and_evolve_field(
        self,
        context: Dict[str, Any],
        field: DataField,
        current_record: Dict[str, Any],
    ) -> Any:
        field_value = self._generate_field_value(context, field, current_record)
        self.logger.debug(f"Initial value for {field.name}: {field_value}")

        evolution_rate = field.evolution_rate

        for gen in range(self.model_definition.evol_generations):
            if random.random() < evolution_rate:
                evolution_strategy = self._select_evolutionary_strategy(field)
                if evolution_strategy:
                    self.logger.info(
                        f"Applying evolution strategy '{evolution_strategy}' to {field.name}"
                    )
                    evolved_value = self._mutate_field_value(
                        field_value,
                        evolution_strategy,
                        field,
                        context,
                        current_record=current_record,
                    )
                    field_value, _ = self._validate_and_correct_field_value(
                        evolved_value, field
                    )
            else:
                self.logger.debug(
                    f"No evolution applied to {field.name} in generation {gen+1}"
                )

        return field_value

    def _setup_logging(self):
        """Set up logging for the EvolDataGenerator."""
        logging.basicConfig(
            level=getattr(logging, self.model_definition.log_level),
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        logging.getLogger("sqlfluff").setLevel(logging.WARNING)

    def _initialize_validators(self) -> Dict[str, Callable]:
        validators = {}
        for field in self.model_definition.fields:
            if field.validator:
                validators[field.name] = self._get_validator(field.validator)
        return validators

    def _get_validator(self, validator_type: str) -> Callable:
        """
        Returns a validation function based on the validator type provided.

        Supported validators:
            - sql: SQL validation
            - json: JSON validation
            - python: Python code validation
            - Custom: Any other validator will be handled by the LLM as a custom validator.

        If an unsupported validator type is specified, the LLM will attempt to validate
        the content based on the provided type description.
        """
        validator_map = {
            "sql": ContentValidator.validate_sql,
            "json": ContentValidator.validate_json,
            "python": ContentValidator.validate_python,
        }
        validator_name, *params = validator_type.split(":")
        validator = validator_map.get(validator_name)

        if validator:
            self.logger.info(f"Using pre-built validator for {validator_name}.")
            return lambda content: validator(content, validator_name, *params)
        else:
            self.logger.info(f"Using LLM-based custom validator for {validator_type}.")
            return lambda content: self._llm_validate(content, validator_type)

    def _llm_validate(self, content: str, content_type: str) -> Optional[str]:
        prompt = textwrap.dedent(
            f"""Validate if the following content is valid {content_type}:
            {content}
            If it's valid, return 'VALID'. If not, describe the error."""
        )

        response = self.llm.generate(prompt, temperature=0.2, max_tokens=100)
        if "VALID" in response:
            return None
        return response.strip()

    def _process_contextual_tags(self, num_examples: int) -> List[Dict[str, Any]]:
        if self.model_definition.contextual_tags is None:
            self.logger.info("No contextual tags provided. Using empty context.")
            return [{}] * num_examples

        self.logger.info("Processing contextual tags...")
        contexts = self.model_definition.contextual_tags.mix_tags(num_examples)
        self.logger.info(f"Generated {len(contexts)} contexts from contextual tags.")
        return contexts

    def _process_data_source(self, num_examples: int) -> List[Dict[str, Any]]:
        if self.model_definition.data_source:
            self.logger.info(
                f"Loading data from {self.model_definition.data_source.uri}"
            )
            data = self.model_definition.load_data()
            return self.model_definition.sample_data(num_examples)
        elif self.model_definition.contextual_tags:
            self.logger.info("Processing contextual tags...")
            return self.model_definition.contextual_tags.mix_tags(num_examples)
        else:
            self.logger.info(
                "No data source or contextual tags provided. Using empty context."
            )
            return [{}] * num_examples

    def generate_data(self) -> Generator[Dict[str, Any], None, str]:
        """
        Generates data records by processing the context, generating fields,
        validating the records, and performing a judge check on each record.
        Valid records are saved to an output file in JSONL format.

        The method uses model definitions and fields from `self.model_definition` to
        create records. It logs the progress and results and updates success and failure counts.

        Yields:
            Dict[str, Any]: A valid merged record combining the context and generated fields.
        Returns:
            str: The final filename that was used for writing the data.
        """
        num_examples = self.model_definition.num_examples
        base_output_filename = self.model_definition.output_filename
        contexts = self._process_data_source(num_examples)

        # Ensure the filename is unique by appending a counter if necessary
        self.output_filename = base_output_filename
        counter = 1
        while os.path.exists(self.output_filename):
            self.output_filename = f"{base_output_filename}_{counter}.jsonl"
            counter += 1

        self.logger.info(f"Output file: {self.output_filename}")

        # Open the output file in write mode to clear existing content
        with open(self.output_filename, "w") as f:
            pass  # Just to clear the file if it exists

        with open(self.output_filename, "a") as f:
            for i, context in enumerate(contexts):
                self.logger.info(f"Generating record {i+1}/{num_examples}")
                self.logger.debug(f"Context: {json.dumps(context, indent=2)}")
                record = {}
                valid_record = True

                # Generate fields for each record
                for field in self.model_definition.fields:
                    self.logger.info(f"Generating field: {field.name}")
                    field_value = self._generate_and_evolve_field(
                        context, field, record
                    )
                    field_value, is_valid = self._validate_and_correct_field_value(
                        field_value, field
                    )

                    if not is_valid:
                        valid_record = False
                        self.fail_count += 1
                        break

                    record[field.name] = field_value

                # If the record is valid, proceed to judge check
                if valid_record:
                    passed_judge, judge_response = self._llm_judge_check(record)
                    if passed_judge:
                        # Combine the context and the generated record
                        merged_record = {**context, **record}
                        self._print_record(merged_record)
                        self.success_count += 1

                        # Write the valid record to the output file
                        json.dump(merged_record, f)
                        f.write("\n")  # Ensure newline after each record
                        self.logger.debug(
                            f"Record passed LLM judge check: {judge_response}"
                        )

                        # Yield the record
                        yield merged_record
                    else:
                        self.logger.warning(
                            f"Record failed LLM judge check and was dropped: {judge_response}\n\n{record}\n\n"
                        )
                        self.fail_count += 1
                else:
                    self.logger.warning("Record failed validation and was dropped")

                # Print the stats after each record generation
                self.logger.info(
                    f"Stats: {self.success_count} successful generations, {self.fail_count} failed generations"
                )

    def _create_field_prompt(
        self,
        context: Dict[str, Any],
        field: DataField,
        current_record: Dict[str, Any],
    ) -> str:
        return textwrap.dedent(
            f"""
            {self.model_definition.generation_instructions}

            Context:
            {json.dumps(context, indent=2)}

            Current record:
            {json.dumps(current_record, indent=2)}

            Generate a value for the following field:
            Name: {field.name}
            Type: {field.type}
            Description: {field.description}

            Ensure the generated value is consistent with the context and current record.
            """
        )

    def _parse_field_value(self, value: str, field: DataField) -> Any:
        if field.type == "int":
            return int(value)
        elif field.type == "float":
            return float(value)
        return value

    def _print_record(self, record: Dict[str, Any]):
        print("\nGenerated Record:")
        print("=" * 50)
        for field_name, field_value in record.items():
            print(f"\n{field_name.upper()}:")
            print("-" * 50)
            if isinstance(field_value, str) and (
                "\n" in field_value or len(field_value) > 80
            ):
                print(f"'''\n{field_value}\n'''")
            else:
                print(field_value)
        print("=" * 50)
