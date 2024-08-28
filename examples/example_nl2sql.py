import textwrap
import pandas as pd

from navigator_helpers import (
    DataFieldDefinition,
    DataModelDefinition,
    GeneratorConfig,
    SyntheticDataGenerator,
)


def main():
    # Define the configuration
    config = GeneratorConfig(
        api_key="prompt",
        llm_model="gretelai/gpt-auto",
        num_generations=1,
        log_level="INFO",
    )

    model_def = DataModelDefinition(
        system_message=textwrap.dedent("""You are a seasoned SQL expert specializing in crafting intricate, context-rich queries and explanations. 
        * Use the provided contextual tags as instructions for generation. Reference the provided Context to ensure relevance and appropriate complexity.
        """),
        fields=[
            DataFieldDefinition(
                name="sql_context",
                type="str",
                description="A single string comprising multiple valid PostgreSQL `CREATE TABLE` statements and a complex schema similar to a production application including multiple tables, separated by semicolons. The schema should be based on the provided Context, particularly the domain and domain_description.",
                validator="sql:postgres",
                mutation_strategies=["complexity", "improve"],
            ),
            DataFieldDefinition(
                name="prompt",
                type="str",
                description="A detailed, nuanced natural language question related to SQL and databases, based on the provided `sql_context` field that challenges advanced understanding. The prompt should align with the domain and domain_description from the contextual tags.",
                validator="English",
                mutation_strategies=["diversity", "complexity", "improve"],
            ),
            DataFieldDefinition(
                name="sql",
                type="str",
                description="A fully executable SQL query that directly answers the `prompt` using the schema in `sql_context`, with no markup or extraneous explanations. The query complexity should match the sql_complexity specified in the contextual tags.",
                validator="sql:postgres",
                mutation_strategies=["complexity", "improve"],
            ),
            DataFieldDefinition(
                name="sql_explanation",
                type="str",
                description="A comprehensive step-by-step breakdown of the SQL query, detailing how it answers the `prompt` and the purpose of each part. Include references to the domain-specific context.",
                mutation_strategies=["simplify", "improve"],
            ),
        ],
    )

    # Create or load contextual tags for diversity
    contextual_tags = pd.read_csv("./docs/data/sql_contextual_tags.csv")

    # Initialize the SyntheticDataGenerator
    generator = SyntheticDataGenerator(config, model_def)

    # Generate the data
    synthetic_data = generator.generate_data(
        contextual_tags, output_file="output.jsonl"
    )
    print("Synthetic data generation complete.")


if __name__ == "__main__":
    main()
