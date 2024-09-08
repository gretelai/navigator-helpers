"""
This script generates synthetic data for NL2SQL tasks.

It uses an LLM to create natural language questions based on complex SQL schemas, 
and then generates corresponding SQL queries and detailed explanations.
"""

import textwrap

import pandas as pd

from navigator_helpers import (
    DataFieldDefinition,
    DataModelDefinition,
    EvolDataGenerator,
    GeneratorConfig,
)


def main():
    # Define the configuration
    config = GeneratorConfig(
        api_key="prompt",
        llm_model="gretelai/gpt-auto",
        num_generations=1,
        log_level="INFO",
        use_reflection=True,
    )

    model_def = DataModelDefinition(
        generation_instructions=textwrap.dedent(
            """You are a seasoned SQL expert specializing in crafting intricate, context-rich queries and explanations. 
        * Use the provided contextual tags as instructions for generation. Reference the provided Context to ensure relevance and appropriate complexity.
        """
        ),
        fields=[
            DataFieldDefinition(
                name="sql_context",
                type="str",
                description="An executable SQL query containing multiple valid PostgreSQL CREATE TABLE statements that define a complex schema, resembling a production environment. The schema includes multiple interrelated tables, separated by semicolons, and should be generated without any additional markup. The structure must align with the provided context, especially the specified domain and its description.",
                validator="sql:postgres",
                evolution_strategies=["complexity", "improve"],
                evolution_rate=0.0,
            ),
            DataFieldDefinition(
                name="prompt",
                type="str",
                description="A detailed, nuanced natural language question related to SQL and databases, based on the provided `sql_context` field that challenges advanced understanding. The prompt should align with the domain and domain_description from the contextual tags.",
                validator="A natural language question or command written in English",
                evolution_strategies=["diversity", "complexity", "improve"],
                evolution_rate=0.0,
            ),
            DataFieldDefinition(
                name="sql",
                type="str",
                description="A fully executable SQL query that directly answers the `prompt` using the schema in `sql_context`, with no markup or extraneous explanations. The query complexity should match the sql_complexity specified in the contextual tags.",
                validator="sql:postgres",
                evolution_strategies=["complexity", "improve"],
                evolution_rate=0.0,
            ),
            DataFieldDefinition(
                name="sql_explanation",
                type="str",
                description="A comprehensive step-by-step breakdown of the SQL query, detailing how it answers the `prompt` and the purpose of each part. Include references to the domain-specific context.",
                validator="A natural language explanation written in English",
                evolution_strategies=["simplify", "improve"],
                evolution_rate=0.0,
            ),
        ],
    )

    # Create or load contextual tags for diversity
    contextual_tags = pd.read_csv("./docs/data/sql_contextual_tags.csv")

    # Initialize the SyntheticDataGenerator
    generator = EvolDataGenerator(config, model_def)

    # Generate the data
    synthetic_data = generator.generate_data(
        contextual_tags, output_file="nl2sql_synthetic_data.jsonl"
    )
    print("NL2SQL data generation complete.")


if __name__ == "__main__":
    main()
