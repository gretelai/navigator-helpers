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
        self.gretel = Gretel(api_key=self.config.api_key)
        self.llm = self.gretel.factories.initialize_navigator_api(
            "natural_language", backend_model=self.config.llm_model
        )
        self._setup_logging()
        self.validators = self._initialize_validators()
        self.evolutionary_strategies = self._initialize_evolution_strategies()

    def _initialize_evolution_strategies(self) -> Dict[str, List[str]]:
        default_strategies = get_prebuilt_evolutionary_strategies()
        return {**default_strategies, **self.custom_evolutionary_strategies}

    def _select_evolutionary_strategy(self, field: DataFieldDefinition) -> str:
        if not field.evolutionary_strategies:
            self.logger.warning(
                f"No evolution strategies defined for field {field.name}. Skipping evolution."
            )
            return ""
        category = random.choice(list(field.evolutionary_strategies))
        return random.choice(self.evolutionary_strategies[category])

    def _generate_field_value(
        self,
        context: Dict[str, Any],
        field: DataFieldDefinition,
        current_record: Dict[str, Any],
    ) -> Any:
        prompt = self._create_field_prompt(context, field, current_record)
        self.logger.debug(f"Prompt for {field.name}:\n{prompt}")
        response = self.llm.generate(prompt, temperature=0.7, max_tokens=MAX_TOKENS)
        value = self._parse_field_value(response.strip(), field)
        self.logger.debug(f"Generated value for {field.name}: {value}")
        return value

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

    def _mutate_field_value(
        self,
        value: Any,
        evolution_strategy: str,
        field: DataFieldDefinition,
        context: Dict[str, Any],
        current_record: Dict[str, Any],
    ) -> Any:
        # Prepare the context from already generated fields
        context_json = json.dumps(context)
        current_record_json = json.dumps(current_record, indent=2)

        # Build the evolution prompt
        evolution_prompt = textwrap.dedent(
            f"""
        Apply the following evolution strategy to the given value:
        Strategy: {evolution_strategy}
        Current value: {value}
        Field type: {field.type}
        Field description: {field.description}
        Context: {context_json}
        Current Record (already generated fields): {current_record_json}

        Ensure that the evolved value remains consistent with the fields that have already been generated in the current record.
        Return only the evolved value.
        """
        )

        self.logger.debug(f"evolution prompt for {field.name}:\n{evolution_prompt}")

        # Generate the evolved value
        evolved_response = self.llm.generate(
            evolution_prompt, temperature=0.8, max_tokens=MAX_TOKENS
        )

        # Parse the evolved value
        evolved_value = self._parse_field_value(evolved_response.strip(), field)

        self.logger.debug(f"evolved value for {field.name}: {evolved_value}")
        return evolved_value

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
            If it's valid, return 'Valid'. If not, describe the error."""
        )

        response = self.llm.generate(prompt, temperature=0.2, max_tokens=100)
        if response.strip().lower().startswith("valid"):
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

    def _validate_and_correct_field_value(
        self, value: Any, field: DataFieldDefinition
    ) -> Tuple[Any, bool]:
        validator = self.validators.get(field.name)
        if validator:
            error = validator(value)
            if error:
                self.logger.warning(f"Validation error for {field.name}: {error}")
                corrected_value = self._correct_content(value, field.type, error)
                if corrected_value != value:
                    self.logger.info(f"Repaired value for {field.name}")
                    return corrected_value, True
                else:
                    self.logger.warning(f"Failed to repair value for {field.name}")
                    return value, False
        return value, True

    def _correct_content(
        self, content: str, content_type: str, error_message: str
    ) -> str:
        prompt = textwrap.dedent(
            f"""The following {content_type} content is invalid:
        Error: {error_message}
        Please correct it so that it conforms to valid {content_type.upper()} syntax.
        {content}
        Return only the corrected version of the provided content, with no additional text, explanations, or formatting."""
        )
        response = self.llm.generate(prompt, temperature=0.2, max_tokens=MAX_TOKENS)
        return response.strip()

    def _llm_judge_check(self, generated_record: Dict[str, Any]) -> Tuple[bool, str]:
        prompt = textwrap.dedent(
            f"""
        As an expert judge, your task is to evaluate the quality and relevance of the following generated record:

        {json.dumps(generated_record, indent=2)}

        Based on the original data model definition:

        {self.model_definition.model_dump_json(indent=2)}

        Evaluate the generated record on the following criteria:
        1. Adherence to instructions in the system message
        2. Relevance to the specified fields and their descriptions
        3. Consistency with the provided context
        4. Overall quality and coherence

        If the record meets all criteria, respond with "PASS" followed by a brief explanation.
        If the record fails to meet any criteria, respond with "FAIL" followed by a explanation of why it failed.

        Your response:
        """
        )

        response = self.llm.generate(prompt, temperature=0.2, max_tokens=MAX_TOKENS)

        if response.strip().startswith("PASS"):
            return True, response.strip()
        else:
            return False, response.strip()

    def generate_data(
        self,
        contextual_tags: Optional[pd.DataFrame] = None,
        output_file: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        contexts = self._process_contextual_tags(contextual_tags)

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
                    break
                record[field.name] = field_value

            if valid_record:
                passed_judge, judge_response = self._llm_judge_check(record)
                if passed_judge:
                    self._print_record(record)
                    results.append(record)
                    if output_file:
                        self._write_to_output(record, output_file)
                    self.logger.debug(
                        f"Record passed LLM judge check: {judge_response}"
                    )
                else:
                    self.logger.warning(
                        f"Record failed LLM judge check and was dropped: {judge_response}"
                    )
            else:
                self.logger.warning("Record failed validation and was dropped")

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
