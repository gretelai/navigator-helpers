"""
This script generates synthetic data for NL2SQL tasks.
It uses an LLM to create natural language questions based on complex SQL schemas, 
and then generates corresponding SQL queries and detailed explanations.
"""

import textwrap

from typing import Dict, List

import pandas as pd

from navigator_helpers import (
    DataFieldDefinition,
    DataModelDefinition,
    EvolDataGenerator,
    GeneratorConfig,
)

# Constants
CONTEXTUAL_TAGS_FILE = "./docs/data/sql_contextual_tags.csv"
OUTPUT_FILE = "nl2sql_synthetic_data.jsonl"


def create_model_definition() -> DataModelDefinition:
    """Creates and returns the DataModelDefinition for NL2SQL tasks."""
    return DataModelDefinition(
        generation_instructions=textwrap.dedent(
            """You are a seasoned SQL expert specializing in crafting intricate, context-rich queries and explanations. 
            * Use the provided contextual tags as instructions for generation. Reference the provided Context to ensure relevance and appropriate complexity.
            * Ensure all generated SQL is valid PostgreSQL syntax.
            * Create realistic and complex database schemas that reflect real-world applications.
            * Generate natural language prompts that challenge SQL expertise and reflect real business needs.
            * Provide detailed, step-by-step explanations of the SQL queries, including rationale for design choices.
            """
        ),
        fields=[
            DataFieldDefinition(
                name="sql_context",
                type="str",
                description="A single string comprising multiple valid PostgreSQL `CREATE TABLE` statements and a complex schema similar to a production application including multiple tables, separated by semicolons. The schema should be based on the provided Context, particularly the domain and domain_description.",
                validator="sql:postgres",
                evolution_strategies=[
                    "Enhance the schema to include domain-specific tables and data types.",
                    "Add relevant indexes, constraints, and views reflecting real-world designs.",
                ],
                evolution_rate=0.1,
            ),
            DataFieldDefinition(
                name="prompt",
                type="str",
                description="A detailed, nuanced natural language prompt that a user might ask to a database for a particular task, based on the provided `sql_context` field that challenges advanced understanding. The prompt should align with the domain and domain_description from the contextual tags.",
                validator="A natural language question or command written in English",
                evolution_strategies=[
                    "Refine the prompt to sound more natural.",
                    "Ensure the prompt reflects real-world business needs.",
                ],
                evolution_rate=0.1,
            ),
            DataFieldDefinition(
                name="sql",
                type="str",
                description="A fully executable SQL query that directly answers the `prompt` using the schema in `sql_context`, with no markup or extraneous explanations. The query complexity should match the sql_complexity specified in the contextual tags.",
                validator="sql:postgres",
                evolution_strategies=[
                    "Optimize the SQL solution for performance.",
                    "Ensure the solution handles large datasets efficiently.",
                ],
                evolution_rate=0.1,
            ),
            DataFieldDefinition(
                name="sql_explanation",
                type="str",
                description="A comprehensive step-by-step breakdown of the SQL query, detailing how it answers the `prompt` and the purpose of each part. Include references to the domain-specific context.",
                evolution_strategies=[
                    "Provide a clear, step-by-step explanation.",
                    "Explain the rationale behind specific SQL techniques.",
                ],
                evolution_rate=0.1,
            ),
        ],
    )

def load_contextual_tags() -> pd.DataFrame:
    """Loads contextual tags from a CSV file."""
    return pd.read_csv(CONTEXTUAL_TAGS_FILE)


def main():
    """Main function to generate synthetic NL2SQL data."""
    print("Starting NL2SQL synthetic data generation...")

    # Set up generator configuration
    config = GeneratorConfig(
        api_key="prompt",
        llm_model="gretelai/gpt-auto",
        num_generations=1,
        log_level="INFO",
        use_reflection=True,
    )

    # Load contextual tags
    contextual_tags = load_contextual_tags()

    # Initialize the SyntheticDataGenerator
    generator = EvolDataGenerator(
        config,
        create_model_definition(),
    )

    # Generate the data
    synthetic_data = generator.generate_data(contextual_tags, output_file=OUTPUT_FILE)

    print(f"NL2SQL data generation complete. Output saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
