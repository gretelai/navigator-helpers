import json
import logging
import random
import textwrap

from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from gretel_client import Gretel

from .content_validator import ContentValidator
from .data_models import DataFieldDefinition, DataModelDefinition, GeneratorConfig
from .evolutionary_strategies import get_prebuilt_evolutionary_strategies
from .text_inference import TextInference
from .prompts import (
    LLM_JUDGE_PROMPT,
    CONTENT_CORRECTION_PROMPT,
    FIELD_GENERATION_PROMPT,
    MUTATION_PROMPT,
)

MAX_TOKENS = 2048  # Max generation tokens


class EvolDataGenerator:
    def __init__(
        self,
        config: GeneratorConfig,
        model_definition: DataModelDefinition,
        custom_evolutionary_strategies: Optional[Dict[str, List[str]]] = None,
    ):
        self.config = config
        self.model_definition = model_definition
        self.custom_evolutionary_strategies = custom_evolutionary_strategies or {}
        self._setup_logging()
        self.gretel = Gretel(api_key=self.config.api_key)
        self.llm = self.gretel.factories.initialize_navigator_api(
            "natural_language", backend_model=self.config.llm_model
        )
        self.text_inference = TextInference(self.llm, self.logger)
        self.use_reflection = config.use_reflection
        self.validators = self._initialize_validators()
        self.evolutionary_strategies = self._initialize_evolution_strategies()

        self.success_count = 0
        self.fail_count = 0

    def _initialize_evolution_strategies(self) -> Dict[str, List[str]]:
        default_strategies = get_prebuilt_evolutionary_strategies()
        return {**default_strategies, **self.custom_evolutionary_strategies}

    def _select_evolutionary_strategy(self, field: DataFieldDefinition) -> str:
        if not field.evolution_strategies:
            self.logger.warning(
                f"No evolution strategies defined for field {field.name}. Skipping evolution."
            )
            return ""
        category = random.choice(list(field.evolution_strategies))
        return random.choice(self.evolutionary_strategies[category])

    def _generate_field_value(
        self,
        context: Dict[str, Any],
        field: DataFieldDefinition,
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
        field: DataFieldDefinition,
        context: Dict[str, Any],
        current_record: Dict[str, Any],
    ) -> Any:
        prompt = MUTATION_PROMPT.format(
            generation_instructions=self.model_definition.generation_instructions,
            evolution_strategy=evolution_strategy,
            value=value,
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
        self, value: Any, field: DataFieldDefinition
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
        field: DataFieldDefinition,
    ) -> str:

        return self.text_inference.generate(
            CONTENT_CORRECTION_PROMPT.format(
                model_definition=self.model_definition.model_dump_json(indent=2),
                content_type=content_type.upper(),
                error_message=error_message,
                content=content,
            ),
            use_reflection=self.config.use_reflection,
            return_full_reflection=field.store_full_reflection,
            temperature=0.2,
            field_name="Attempting Correction",
        )

    def _generate_and_evolve_field(
        self,
        context: Dict[str, Any],
        field: DataFieldDefinition,
        current_record: Dict[str, Any],
    ) -> Any:
        field_value = self._generate_field_value(context, field, current_record)
        self.logger.debug(f"Initial value for {field.name}: {field_value}")

        evolution_rate = (
            field.evolution_rate
            if field.evolution_rate is not None
            else self.config.evolution_rate
        )

        for gen in range(self.config.num_generations):
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
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
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

    def _process_contextual_tags(
        self, contextual_tags: Optional[pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        if contextual_tags is None:
            return [{}]  # Return a list with one empty context if no tags provided
        return contextual_tags.to_dict("records")

    def _create_field_prompt(
        self,
        context: Dict[str, Any],
        field: DataFieldDefinition,
        current_record: Dict[str, Any],
    ) -> str:
        return textwrap.dedent(
            f"""
            {self.model_definition.system_message}

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

    def _parse_field_value(self, value: str, field: DataFieldDefinition) -> Any:
        if field.type == "int":
            return int(value)
        elif field.type == "float":
            return float(value)
        # Add more type parsing as needed
        return value

    def generate_data(
        self,
        contextual_tags: Optional[pd.DataFrame] = None,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        contexts = self._process_contextual_tags(contextual_tags)

        # Open the output file in write mode to overwrite it each time
        if output_file:
            with open(output_file, "w") as f:
                f.write("")

        for i, context in enumerate(contexts):
            self.logger.info(f"Generating record {i+1}/{len(contexts)}")
            self.logger.debug(f"Context: {json.dumps(context, indent=2)}")

            record = {}
            valid_record = True
            for field in self.model_definition.fields:
                self.logger.info(f"Generating field: {field.name}")
                field_value = self._generate_and_evolve_field(context, field, record)
                field_value, is_valid = self._validate_and_correct_field_value(
                    field_value, field
                )
                if not is_valid:
                    valid_record = False
                    self.fail_count += 1
                    break
                record[field.name] = field_value

            if valid_record:
                passed_judge, judge_response = self._llm_judge_check(record)
                if passed_judge:
                    # Combine the context and the generated record
                    merged_record = {**context, **record}
                    self._print_record(merged_record)
                    results.append(merged_record)

                    self.success_count += 1
                    if output_file:
                        self._write_to_output(merged_record, output_file)
                    self.logger.debug(
                        f"Record passed LLM judge check: {judge_response}"
                    )
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

        return results

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

    def _write_to_output(self, record: Dict[str, Any], output_file: str):
        with open(output_file, "a") as f:
            json.dump(record, f)
            f.write("\n")
