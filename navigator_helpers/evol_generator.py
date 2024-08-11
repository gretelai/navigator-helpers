import ast
import json
import logging
import random
import re
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import sqlfluff
from gretel_client import Gretel
from tqdm.auto import tqdm

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("sqlfluff").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_quality_scores(response: str) -> Dict[int, int]:
    pattern = r"Example (\d+): ([0-9.]+)"
    return {int(num): int(float(score)) for num, score in re.findall(pattern, response)}


class ContentValidator:
    @staticmethod
    def validate_sql(
        content: str, content_type: str, dialect: str = "ansi"
    ) -> Optional[str]:
        try:
            result = sqlfluff.lint(content, dialect=dialect)
            prs_errors = [res for res in result if res["code"].startswith("PRS")]
            if prs_errors:
                error_messages = "\n".join(
                    [f"{error['code']}: {error['description']}" for error in prs_errors]
                )
                return error_messages
            return None
        except Exception as e:
            return f"Exception during SQL parsing: {str(e)[:50]}..."

    @staticmethod
    def validate_json(content: str, content_type: str) -> Optional[str]:
        try:
            json.loads(content)
            return None
        except json.JSONDecodeError as e:
            return str(e)

    @staticmethod
    def validate_python(content: str, content_type: str) -> Optional[str]:
        try:
            ast.parse(content)
            return None
        except SyntaxError as e:
            return str(e)


class EvolDataGenerator:
    def __init__(self, config: Dict[str, Any], output_file: str):
        self.config = config
        self.output_file = output_file
        self.gretel = Gretel(api_key=config["api_key"])
        self.tabular = self.gretel.factories.initialize_navigator_api(
            backend_model=config["tabular_model"]
        )
        self.llm = self.gretel.factories.initialize_navigator_api(
            "natural_language", backend_model=config["llm_model"]
        )
        self.validated_fields: Dict[str, str] = {}
        self.content_validator = ContentValidator()
        self.column_validators = self._initialize_validators()
        logger.info(
            f"EvolDataGenerator initialized with configuration:\n{json.dumps(config, indent=2)}"
        )
        self.mutation_strategies = self._initialize_mutation_strategies()
        self.records_filtered = 0
        self.records_repaired = 0
        self.relevance_checks_failed = 0
        self.safety_checks_failed = 0

    def _initialize_mutation_strategies(self) -> Dict[str, List[str]]:
        return {
            "improve": [
                "Improve the content, enhancing its quality, clarity, coherence, or effectiveness while maintaining its core meaning.",
                "Optimize the content for better performance, efficiency, or clarity, rewriting it to be more concise, effective, or clear.",
            ],
            "simplify": [
                "Simplify the content to make it more accessible and understandable.",
                "Make the content more abstract or generalized, broadening the scope to cover a wider range of scenarios or applications.",
            ],
            "complexity": [
                "Increase the complexity and nuance of the content by introducing more sophisticated concepts, layers of detail, or intricate relationships.",
                "Expand the content by adding more details, explanations, or edge cases, such as considering uncommon scenarios, edge conditions, or additional attributes.",
            ],
            "diversity": [
                "Provide an alternative way to solve or express the same problem or concept, introducing different approaches, perspectives, or methods.",
                "Adapt the content to a closely related context within the same domain, ensuring that the core focus remains aligned with the original intent.",
                "Make the content more specific or concrete with particular details, focusing on a narrower, more detailed aspect of the original prompt.",
                "Rewrite the content in a different style or format, such as changing the tone, structure, or presentation.",
                "Present the content from a different perspective or point of view, potentially introducing new dimensions, angles, or considerations.",
            ],
        }

    def _select_mutation_strategy(self) -> str:
        selected_categories = self.config.get(
            "mutation_categories", list(self.mutation_strategies.keys())
        )
        selected_strategies = []
        for category in selected_categories:
            selected_strategies.extend(self.mutation_strategies.get(category, []))

        if not selected_strategies:
            logger.warning(
                "No valid mutation strategies selected. Using all strategies."
            )
            selected_strategies = [
                strategy
                for strategies in self.mutation_strategies.values()
                for strategy in strategies
            ]

        return random.choice(selected_strategies)

    def _initialize_validators(self) -> Dict[str, Callable]:
        validator_map = {
            "sql": self.content_validator.validate_sql,
            "json": self.content_validator.validate_json,
            "python": self.content_validator.validate_python,
        }

        validators = {}
        for column, validator_type in self.config.get("column_validators", {}).items():
            validator_type_parts = validator_type.split(":")
            base_validator = validator_type_parts[0].lower()
            params = validator_type_parts[1:] if len(validator_type_parts) > 1 else []

            if base_validator in validator_map:
                if base_validator == "sql":
                    dialect = params[0] if params else "ansi"
                    validators[column] = (
                        lambda content, content_type: validator_map[base_validator](
                            content, content_type, dialect=dialect
                        ),
                        base_validator,
                    )
                else:
                    validators[column] = (validator_map[base_validator], base_validator)
            else:
                validators[column] = (self.llm_validate, base_validator)

            self.validated_fields[column] = (
                f"{base_validator}:{':'.join(params)}" if params else base_validator
            )

            logger.debug(
                f"Configured {base_validator} validator for column '{column}' with parameters: {params}"
            )

        return validators

    def validate_content(self, content: str, content_type: str) -> Optional[str]:
        validator = self.column_validators.get(content_type)
        if not validator:
            logger.warning(
                f"No validator found for content type '{content_type}', defaulting to LLM validation."
            )
            return self.llm_validate(content, content_type)
        return validator(content, content_type)

    def llm_validate(self, content: str, content_type: str) -> Optional[str]:
        prompt = f"""Parse the following content and verify that it conforms to the {content_type} syntax:

{content}

If it's valid {content_type} syntax, respond with 'VALID'. If it's invalid, respond with 'INVALID: ' followed by a brief explanation."""

        response = self.llm.generate(prompt, temperature=0.2)

        if response.strip().startswith("VALID"):
            return None
        return response.strip()[8:]  # Remove 'INVALID: ' prefix

    def validate_and_correct_data(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting data validation and correction process")
        for column, (validator_func, validator_type) in self.column_validators.items():
            if column in data.columns:
                data[column] = data[column].apply(
                    lambda x: self.validate_and_correct_content(
                        x, validator_func, validator_type
                    )
                )
        logger.info("Data validation and correction completed")
        return data

    def validate_and_correct_content(
        self, content: str, validator_func: Callable, content_type: str
    ) -> str:
        error = validator_func(content, content_type)
        if error:
            logger.warning(f"Validation error for {content_type}: {error}")
            logger.error(f"INVALID content detected: {content[:100]}...")
            corrected_content, explanation = self.correct_content(
                content, content_type, error
            )
            logger.info(
                f"Corrected content of type {content_type}. Explanation: {explanation}"
            )
            self.records_repaired += 1
            return corrected_content
        return content

    def correct_content(
        self, content: str, content_type: str, error_message: str
    ) -> tuple[str, str]:
        logger.debug(f"Correcting {content_type} content using structured generation")

        expected_columns = ["corrected_content", "explanation"]

        prompt = f"""The following {content_type} content is invalid:
Error: {error_message}
Please correct it so that it conforms to valid {content_type.upper()} syntax.

{content}

Return a dataset with the following columns:
- `corrected_content`: Only the corrected version of the provided content, with no additional text, explanations, or formatting.
- `explanation`: A brief explanation of the corrections made, if applicable."""

        response = self.safe_tabular_generate(
            prompt=prompt, num_records=1, expected_columns=expected_columns
        )

        corrected_content = response["corrected_content"].iloc[0]
        explanation = response["explanation"].iloc[0]
        logger.debug(f"Correction applied for {content_type}. {explanation}")
        return corrected_content, explanation

    def _create_column_list(self) -> str:
        return "\n".join(
            [f"- `{col}`" for col in self.columns if col != "quality_score"]
        )

    def _create_prompt(self, user_prompt: str, context_row: pd.Series) -> str:
        input_fields = self.config.get("input_fields", context_row.index)
        context = "\n".join(
            f"{key}: {value}"
            for key, value in context_row.items()
            if key in input_fields
        )
        return f"{user_prompt}\n\nContext:\n{context}\n\n"

    def safe_tabular_generate(
        self, prompt: str, num_records: int, expected_columns: List[str]
    ) -> pd.DataFrame:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.tabular.generate(prompt=prompt, num_records=num_records)
                missing_columns = [
                    col for col in expected_columns if col not in response.columns
                ]
                if not missing_columns:
                    return response
                else:
                    raise ValueError(f"Missing columns in response: {missing_columns}")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(
                        f"Max retries reached. Failed to generate response for prompt:\n{prompt}"
                    )
                    raise e
                time.sleep(2)

    def _generate_initial_population(self, prompt: str) -> pd.DataFrame:
        logger.info("Generating initial population")

        expected_columns = self.config.get("expected_columns", [])

        population = self.safe_tabular_generate(
            prompt=prompt,
            num_records=self.config["population_size"],
            expected_columns=expected_columns,
        )

        self.columns = population.columns.tolist()
        logger.info(f"Initial population generated with shape: {population.shape}")

        return self.validate_and_correct_data(population)

    def _expand_population(
        self, population: pd.DataFrame, user_prompt: str
    ) -> pd.DataFrame:
        if self.config.get("expansion_size", 0) == 0:
            logger.info("Expansion size is 0, skipping population expansion.")
            return population

        logger.debug("Expanding population")

        expansion_prompt = (
            f"{user_prompt}\n\n"
            f"Generate new, diverse variations with the following columns:\n{self._create_column_list()}\n\n"
            f"Based on these examples:\n{population.sample(n=(min(len(population), 2))).to_json(orient='records')}"
        )

        expected_columns = [x for x in population.columns if x != "quality_score"]

        expanded = self.safe_tabular_generate(
            prompt=expansion_prompt,
            num_records=self.config["expansion_size"],
            expected_columns=expected_columns,
        )

        logger.info(f"Population expanded. Expansion shape: {expanded.shape}")

        # Combine the new examples with the existing population
        updated_population = pd.concat([population, expanded], ignore_index=True)

        return self.validate_and_correct_data(updated_population)

    def _apply_mutations(
        self, population: pd.DataFrame, user_prompt: str
    ) -> pd.DataFrame:
        if self.config.get("mutation_rate", 0.0) == 0.0:
            logger.info("Mutation rate is 0.0, skipping mutations.")
            return population

        logger.debug("Applying mutations")

        mutated_population = []
        for index, individual in population.iterrows():
            if random.random() < self.config["mutation_rate"]:
                strategy = self._select_mutation_strategy()
                logger.debug(f"Mutating example {index} with strategy: {strategy}")

                expected_columns = [x for x in individual.index if x != "quality_score"]
                columns_str = "\n".join([f"- `{col}`" for col in expected_columns])
                individual_dict = individual.to_dict()
                individual_str = json.dumps(individual_dict, indent=2)
                mutation_prompt = (
                    f"Mutate the data below by applying the following strategy: {strategy}\n"
                    f"Ensure that the mutated data still aligns with the original intent and satisfies the requirements of the original prompt:\n\n{user_prompt}\n\n"
                    f"Return a dataset with the following columns:\n{columns_str}\n"
                    "- `mutation_explanation`: Explain the specific mutations made to the provided data in 1-2 sentences.\n\n"
                    f"{individual_str}\n\n"
                    "For the 'mutation_explanation' column, provide a 1-2 sentence explanation of how the data was mutated and how it aligns with the original intent."
                )

                mutated_individual = self.safe_tabular_generate(
                    prompt=mutation_prompt,
                    num_records=1,
                    expected_columns=expected_columns + ["mutation_explanation"],
                )

                logger.debug(
                    f"Mutation explanation for example {index}: {mutated_individual['mutation_explanation'].iloc[0]}"
                )

                mutated_population.append(
                    mutated_individual.drop(columns=["mutation_explanation"]).iloc[0]
                )
            else:
                logger.debug(f"Example {index} not selected for mutation")
                mutated_population.append(individual)

        # Merge mutated examples into a single dataframe
        mutated = pd.DataFrame(mutated_population, columns=population.columns)

        # Validate and correct the mutated data
        mutated = self.validate_and_correct_data(mutated)
        logger.info(f"Mutations applied. Mutations shape: {mutated.shape}")

        return mutated

    def generate_data(
        self, contextual_tags: pd.DataFrame, user_prompt: str
    ) -> pd.DataFrame:
        logger.info(
            f"Starting data generation for {len(contextual_tags)} contextual tags"
        )
        results = []

        num_generations = max(
            self.config.get("num_generations", 1), 1
        )  # Ensure at least 1 generation

        return_highest_only = self.config.get("return_highest_only", False)

        for record_index, row in tqdm(
            contextual_tags.iterrows(),
            total=len(contextual_tags),
            desc="Generating Data",
        ):
            logger.info(f"Processing record {record_index + 1}/{len(contextual_tags)}")
            prompt = self._create_prompt(user_prompt, row)
            logger.debug(f"Generated prompt with contextual tags:\n{prompt}\n\n")

            # Step 1: Generate initial population
            population = self._generate_initial_population(prompt)
            logger.info(
                f"Record {record_index + 1}: Initial population size: {len(population)}"
            )

            for gen in range(num_generations):
                logger.info(
                    f"Record {record_index + 1}: Starting generation {gen + 1}/{num_generations}"
                )

                # Step 2: Expand population (if applicable)
                if self.config.get("expansion_size", 0) > 0:
                    population = self._expand_population(population, user_prompt)
                    logger.debug(
                        f"Record {record_index + 1}, Generation {gen + 1}: Population size after expansion: {len(population)}"
                    )
                else:
                    logger.info("Skipping population expansion.")

                # Step 3: Apply mutations (if applicable)
                if self.config.get("mutation_rate", 0.0) > 0.0:
                    population = self._apply_mutations(
                        population, user_prompt=user_prompt
                    )
                    logger.debug(
                        f"Record {record_index + 1}, Generation {gen + 1}: Population size after mutation: {len(population)}"
                    )
                else:
                    logger.info("Skipping mutations.")

                # Step 4: Rank the population
                ranked_population = self._rank_population(population, prompt)

                # Step 5: Filter the population
                population = self._filter_population(ranked_population)
                logger.info(
                    f"Record {record_index + 1}, Generation {gen + 1}: Final population size after filtering: {len(population)}"
                )

            # Use only the highest-ranking individual if specified
            if return_highest_only:
                population = population.sort_values(
                    by="quality_score", ascending=False
                ).head(1)
                logger.info(
                    f"Record {record_index + 1}: Returning highest ranking individual with score: {population['quality_score'].iloc[0]}"
                )

            logger.info(
                f"Record {record_index + 1}: Final population size after all generations: {len(population)}"
            )

            # Step 6: Merge contextual tags with the final population
            contextual_data = pd.DataFrame([row] * len(population), columns=row.index)
            combined_data = pd.concat(
                [
                    contextual_data.reset_index(drop=True),
                    population.reset_index(drop=True),
                ],
                axis=1,
            )
            results.append(combined_data)

            with open(self.output_file, "a") as f:
                for _, example in combined_data.iterrows():
                    json.dump(example.to_dict(), f, indent=2)
                    f.write("\n")

        # After the data generation loop
        final_result = pd.concat(results, ignore_index=True)
        logger.info(
            f"Data generation complete. Final result shape: {final_result.shape}"
        )

        # Calculate the total number of records generated, including an estimate for mutations
        total_records_generated = (
            len(contextual_tags)
            * self.config.get("population_size", 1)
            * self.config.get("num_generations", 1)
        )
        estimated_mutations = int(
            total_records_generated
            * self.config.get("mutation_rate", 0.0)
            * self.config.get("num_generations", 1)
        )
        total_records_generated += estimated_mutations

        # Calculate percentages
        percent_repaired = (
            (self.records_repaired / total_records_generated) * 100
            if total_records_generated > 0
            else 0
        )
        percent_filtered = (
            (self.records_filtered / total_records_generated) * 100
            if total_records_generated > 0
            else 0
        )

        # Define a separator for visual clarity
        separator = "=" * 50

        # Concise summary of metrics
        logger.info("")
        logger.info(separator)
        logger.info(f"{'Summary of Metrics':^50}")
        logger.info(separator)
        logger.info(
            f"Total records generated: {total_records_generated} (Mutations: {estimated_mutations})"
        )
        logger.info(
            f"Attempted repairs: {self.records_repaired} ({percent_repaired:.2f}%) | Filtered records: {self.records_filtered} ({percent_filtered:.2f}%)"
        )
        logger.info(
            f"Relevance failures: {self.relevance_checks_failed} | Safety failures: {self.safety_checks_failed}"
        )
        logger.info(separator)
        logger.info("Validated Fields:")
        for field, validator in self.validated_fields.items():
            logger.info(f"  - {field}: {validator}")
        logger.info(separator)
        logger.info("")

        return final_result

    def _rank_population(
        self,
        population: pd.DataFrame,
        original_prompt: str,
        max_retries: int = 3,
        batch_size: int = 3,
    ) -> pd.DataFrame:
        logger.info(
            "Scoring and ranking population for relevance, coherence, factual accuracy, bias, and safety."
        )

        def create_ranking_prompt(
            examples: pd.DataFrame, original_prompt: str, start_index: int
        ) -> str:
            example_prompts = []
            for i, example in enumerate(examples.iterrows(), start=start_index):
                example_prompts.append(f"Example {i}: " + example[1].to_json())
            return f"""Evaluate the following examples based on these criteria:
*  Relevance: How well the example adheres to the original prompt.
*  Coherence: The logical flow, clarity, and internal consistency of the example.
*  Factual Accuracy: The correctness and informativeness of the content.
*  Bias: The absence of unfair prejudice, stereotypes, or favoritism.
*  Safety: The degree to which the content is free from harmful or inappropriate elements.

Original prompt: "{original_prompt}"

Examples to evaluate:
{chr(10).join(example_prompts)}

Provide a quality score for each example on a scale of 1 to 5, where:
1 = Very low quality
2 = Low quality
3 = Average quality
4 = High quality
5 = Very high quality

Format your response as follows:
Example 1: [score]
Example 2: [score]
...

Ensure that your scores reflect meaningful differences between the examples based on the given criteria.
"""

        all_quality_scores = {}

        attempt = 0
        while attempt < max_retries:
            try:
                for start in range(0, len(population), batch_size):
                    batch = population.iloc[start : start + batch_size]
                    ranking_prompt = create_ranking_prompt(
                        batch, original_prompt, start + 1
                    )
                    response = self.llm.generate(ranking_prompt, temperature=0.2)
                    quality_scores = parse_quality_scores(response)

                    if len(quality_scores) != len(batch):
                        raise ValueError(
                            f"Mismatch in number of scores ({len(quality_scores)}) and batch size ({len(batch)})"
                        )

                    # Map the batch indexes to the original population
                    for i, score in enumerate(quality_scores.values()):
                        population_index = start + i
                        all_quality_scores[population_index] = score

                # Resetting index but drop it immediately if it creates a column
                population = population.reset_index(drop=True)
                population["quality_score"] = population.index.map(
                    all_quality_scores.get
                )

                # Check for any NaN values in quality scores
                if population["quality_score"].isna().any():
                    missing_scores = population[population["quality_score"].isna()]
                    logger.error(
                        f"Missing scores for the following examples:\n{missing_scores}"
                    )
                    population["quality_score"].fillna(0, inplace=True)

                ranked_population = population.sort_values(
                    "quality_score", ascending=False
                )
                logger.info(
                    f"Rankings generated. Top 5 quality scores: {ranked_population['quality_score'].head().tolist()}"
                )

                return ranked_population

            except Exception as e:
                logger.warning(
                    f"Error during ranking (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                logger.debug(f"Stack trace:\n{traceback.format_exc()}")
                logger.debug(f"LLM Response:\n{response}")
                logger.debug(f"Quality Scores:\n{all_quality_scores}")
                logger.debug(f"Population DataFrame (reset index):\n{population}")

            attempt += 1
            logger.info(f"Retrying ranking (attempt {attempt + 1}/{max_retries})...")
            time.sleep(2)

        logger.error(
            "Max retries exceeded during ranking. Returning original population without ranking."
        )
        return population.assign(quality_score=3)

    def _filter_population(self, ranked_population: pd.DataFrame) -> pd.DataFrame:
        logger.info("Filtering population")

        quality_filtered = ranked_population[ranked_population["quality_score"] >= 3]
        removed_count = len(ranked_population) - len(quality_filtered)
        self.records_filtered += removed_count
        logger.info(
            f"Filtered out {removed_count} examples ({removed_count/len(ranked_population)*100:.2f}%) with quality score below 3"
        )

        final_filtered = []
        for index, row in quality_filtered.iterrows():
            valid = True
            for column, (
                validator_func,
                validator_type,
            ) in self.column_validators.items():
                if column in row and validator_func != self.llm_validate:
                    validation_error = validator_func(row[column], validator_type)
                    if validation_error:
                        logger.warning(
                            f"Record {index} failed final validation for column '{column}': {validation_error}"
                        )
                        valid = False
                        break

            if valid:
                final_filtered.append(row)

        final_filtered_df = pd.DataFrame(final_filtered).reset_index(drop=True)
        removed_count_strict = len(quality_filtered) - len(final_filtered_df)
        self.records_filtered += removed_count_strict
        logger.info(
            f"Strict validation removed {removed_count_strict} additional examples "
            f"({removed_count_strict/len(quality_filtered)*100:.2f}%)"
        )

        logger.info(
            f"Final population size after strict filtering: {len(final_filtered_df)}"
        )
        return final_filtered_df
