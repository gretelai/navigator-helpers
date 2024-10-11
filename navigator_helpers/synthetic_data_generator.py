import json
import logging
import os
import random

from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from gretel_client import Gretel

from .data_models import DataField, DataModel, GenerationStrategy
from .validator_manager import ValidatorManager
from .record_generation import FieldByFieldGenerator, SingleShotGenerator
from .gretel_text_inference import TextInference
from .llm_eval import LLMEval
from .prompts import DEFAULT_EVOLUTION_STRATEGIES, EVOLUTION_STRATEGY_PROMPT
from .utils.logging import setup_logger


class SyntheticDataGenerator:
    def __init__(
        self,
        model_definition: DataModel,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_definition = model_definition
        self.logger = logger or setup_logger(__name__)
        self.gretel = Gretel(api_key=self.model_definition.api_key)
        self.llm = self.gretel.factories.initialize_navigator_api(
            "natural_language", backend_model=self.model_definition.llm_model
        )
        self.text_inference = TextInference(self.llm, self.logger, debug=False)

        self.field_strategy = FieldByFieldGenerator(
            model_definition, self.text_inference
        )
        self.record_strategy = SingleShotGenerator(
            model_definition, self.text_inference
        )

        self.use_reflection = self.model_definition.use_reflection
        self.evolutionary_strategies = self._initialize_evolution_strategies()
        self.llm_eval = LLMEval(self.text_inference, self.use_reflection)
        self.field_validators = ValidatorManager(
            self.text_inference, self.use_reflection
        )
        self.validators = self._initialize_validators()

        # Metrics
        self.success_count = 0
        self.fail_count = 0

        self.output_filename = None

    def _initialize_evolution_strategies(self) -> List[str]:
        if self.model_definition.evolution.strategies:
            self.logger.info("Using evolutionary strategies from config.")
            return self.model_definition.evolution.strategies
        else:
            self.logger.info("Using default evolutionary strategies.")
            return DEFAULT_EVOLUTION_STRATEGIES

    def _initialize_validators(self) -> Dict[str, Callable]:
        validators = {}
        for field in self.model_definition.fields:
            if field.validator:
                validators[field.name] = self.field_validators.get_validator(
                    field.validator
                )
        return validators

    def _validate_record(self, record: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        valid_record = True
        for field in self.model_definition.fields:
            field_value = record.get(field.name)
            if field_value is None:
                self.logger.warning(
                    f"Field {field.name} is missing in generated record."
                )
                valid_record = False
                break
            field_value, is_valid = self._validate_field(field_value, field)
            if not is_valid:
                valid_record = False
                break
            record[field.name] = field_value

        if valid_record:
            self.logger.info("Starting Eval")
            passed_judge, judge_response = self.llm_eval.check(
                record, self.model_definition
            )
            if passed_judge:
                self.logger.info("LLM Eval Passed")
                valid_record = True
            else:
                self.logger.warning(f"Record Failed LLM Eval: {judge_response}")
                valid_record = False

        return valid_record, record

    def generate_data(self) -> Generator[Dict[str, Any], None, str]:
        num_examples = self.model_definition.num_examples
        base_output_filename = self.model_definition.output_filename
        contexts = self._process_data_source(num_examples)

        self.output_filename = self._get_unique_filename(base_output_filename)
        self.logger.info(f"Output file: {self.output_filename}")

        with open(self.output_filename, "w") as f:
            generated_count = 0
            context_index = 0

            while generated_count < num_examples:
                self.logger.info(
                    f"Generating record {generated_count+1}/{num_examples}"
                )

                if context_index >= len(contexts):
                    context_index = 0

                context = contexts[context_index]
                self.logger.debug(f"Context: {json.dumps(context, indent=2)}")

                # Reset byte counts before generating the record
                self.text_inference.reset_byte_counts()

                # Generate the record using the appropriate strategy
                if (
                    self.model_definition.generation_strategy
                    == GenerationStrategy.RECORD
                ):
                    record = self.record_strategy.generate_record(context)
                else:
                    record = self.field_strategy.generate_record(context)

                # Evolve the record
                record = self._evolve_record(record, context)

                # Validate the record
                valid_record, record = self._validate_record(record)
                merged_record = {**context, **record}

                if valid_record:
                    json.dump(merged_record, f)
                    f.write("\n")
                    self._print_record(merged_record)
                    generated_count += 1
                    self.success_count += 1

                    # Log token metrics for the generated record
                    self._log_token_metrics(merged_record, success=True)

                    # Yield the merged record
                    yield merged_record
                else:
                    self.logger.warning("Record failed validation and was dropped.")
                    self.fail_count += 1

                    # Log token metrics even on failure
                    self._log_token_metrics(merged_record, success=False)

                context_index += 1

                # Log the current success and fail counts
                self.logger.info(
                    f"Current stats - Successes: {self.success_count}, Failures: {self.fail_count}"
                )

        self.logger.info(
            f"Final stats - Successes: {self.success_count}, Failures: {self.fail_count}"
        )
        return self.output_filename

    def _evolve_record(
        self, record: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        if random.random() < self.model_definition.evolution.rate:
            strategy = random.choice(self.evolutionary_strategies)
            self.logger.info(f"Applying evolution strategy: {strategy}")
            evolved_record = self._apply_evolution_strategy(record, context, strategy)
            return evolved_record
        return record

    def _apply_evolution_strategy(
        self, record: Dict[str, Any], context: Dict[str, Any], strategy: str
    ) -> Dict[str, Any]:
        prompt = EVOLUTION_STRATEGY_PROMPT.format(
            strategy=strategy,
            record=self.record_strategy._format_record_as_markdown(record),
            context=self.record_strategy._format_record_as_markdown(context),
        )
        response = self.text_inference.generate(
            prompt,
            use_reflection=self.use_reflection,
            return_full_reflection=False,
            temperature=0.7,
            max_tokens=4000,
            field_name="EVOLVED_EXAMPLE",
        )
        evolved_record = self.record_strategy._parse_markdown_response(response)
        return (
            evolved_record
            if self.record_strategy._validate_record_structure(evolved_record)
            else record
        )

    def _log_token_metrics(self, record: Optional[Dict[str, Any]], success: bool):
        total_bytes_read, total_bytes_written = self.text_inference.get_byte_counts()

        # Calculate output bytes based on the final record
        output_bytes = len(json.dumps(record).encode("utf-8")) if record else 0

        # Calculate token overhead ratio
        if output_bytes > 0:
            overhead_ratio = (total_bytes_read + total_bytes_written) / output_bytes
        else:
            overhead_ratio = 0

        log_message = (
            f"Success: {success}, "
            f"OutputBytes: {output_bytes}, "
            f"LLMBytesRead: {total_bytes_read}, "
            f"LLMBytesWritten: {total_bytes_written}, "
            f"OverheadRatio: {overhead_ratio:.4f}"
        )

        self.logger.info(log_message)

    def _validate_field(self, value: Any, field: DataField) -> Tuple[Any, bool]:
        validator = self.validators.get(field.name)
        if validator:
            error = validator(value)
            if error:
                self.logger.warning(f"Validation error for {field.name}: {error}...")
                corrected_value = self.field_validators.correct_content(
                    value, field.type, error, field, self.model_definition
                )

                if corrected_value != value:
                    new_error = validator(corrected_value)
                    if not new_error:
                        self.logger.info(f"Repaired value for {field.name}")
                        return corrected_value, True
                    else:
                        self.logger.info(
                            f"Failed to repair value for {field.name}. New error: {new_error}"
                        )
                        return corrected_value, False
                else:
                    self.logger.warning(f"Unable to repair value for {field.name}")
                    return value, False
        return value, True

    def _process_data_source(self, num_examples: int) -> List[Dict[str, Any]]:
        if self.model_definition.data_source:
            self.logger.info(
                f"Loading data from {self.model_definition.data_source.uri}"
            )
            return self.model_definition.sample_data(num_examples)
        elif self.model_definition.contextual_tags:
            self.logger.info("Processing contextual tags...")
            return self.model_definition.contextual_tags.mix_tags(num_examples)
        else:
            self.logger.info(
                "No data source or contextual tags provided. Using empty context."
            )
            return [{}] * num_examples

    def _get_unique_filename(self, base_filename: str) -> str:
        filename = base_filename
        counter = 1
        while os.path.exists(filename):
            filename = f"{os.path.splitext(base_filename)[0]}_{counter}{os.path.splitext(base_filename)[1]}"
            counter += 1
        return filename

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

    def get_prompt_bytes(self, data_model: DataModel) -> int:
        """
        Counts bytes of the DataModel, excluding the 'contextual_tags' field.

        Args:
            data_model (DataModel): The data model to be traversed and serialized.

        Returns:
            int: Byte size of the serialized data excluding 'contextual_tags'.
        """
        byte_count = 0

        # Count bytes for generation instructions
        generation_instructions = str(data_model.generation_instructions).encode(
            "utf-8"
        )
        byte_count += len(generation_instructions)

        # Count bytes for each field (name, type, description, validator)
        for field in data_model.fields:
            byte_count += len(str(field.name).encode("utf-8"))
            byte_count += len(str(field.type).encode("utf-8"))
            byte_count += len(str(field.description).encode("utf-8"))
            if field.validator:
                byte_count += len(str(field.validator).encode("utf-8"))

        # Count bytes for evolution section
        byte_count += len(str(data_model.evolution.rate).encode("utf-8"))
        for strategy in data_model.evolution.strategies:
            byte_count += len(str(strategy).encode("utf-8"))

        # Other fields to count (excluding contextual_tags)
        other_fields = data_model.model_dump(
            exclude={
                "fields",
                "generation_instructions",
                "contextual_tags",
                "evolution",
            }
        )
        byte_count += len(json.dumps(other_fields).encode("utf-8"))

        return byte_count
