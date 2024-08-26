import pandas as pd

from navigator_helpers import (
    EvolDataGenerator,
    GeneratorConfig,
    MutationCategory,
    LogLevel,
)

# Define the user prompt
user_prompt = """You are a seasoned SQL expert specializing in crafting intricate, context-rich queries and explanations. Your expertise covers advanced joins, subqueries, window functions, CTEs, and complex aggregations.

**Guidelines:**
- Ensure all SQL follows standard best practices and is executable in common SQL environments.
- Always define `DECIMAL` with both precision and scale.

**Task**:
Generate a dataset with the following columns:
- `sql_prompt`: A detailed, nuanced question related to SQL and databases that challenges advanced understanding.
- `sql_context`: A single string comprising multiple valid SQL `CREATE TABLE` statements, separated by semicolons, representing the necessary database schema for the SQL prompt.
- `sql`: A fully executable SQL query that directly answers the `sql_prompt` using the schema in `sql_context`, with no markup or extraneous explanations.
- `sql_explanation`: A comprehensive step-by-step breakdown of the SQL query, detailing how it answers the `sql_prompt` and the purpose of each part.

Ensure the `sql_context` provides all necessary `CREATE TABLE` statements and enough context to support the generated SQL query and explanation.
"""

# Create or load contextual tags for diversity
contextual_tags = pd.read_csv("./docs/data/sql_contextual_tags.csv")

# Define the configuration
config = GeneratorConfig(
    api_key="prompt",
    tabular_model="gretelai/Llama-3.1-8B-Instruct",
    llm_model="gretelai/gpt-llama3-1-8b",
    num_generations=1,
    population_size=1,
    expansion_size=0,
    mutation_rate=0.6,
    mutation_categories=[
        MutationCategory.COMPLEXITY,
    ],
    expected_columns=["sql_context", "sql", "sql_prompt", "sql_explanation"],
    column_validators={"sql_context": "sql:postgres", "sql": "sql:postgres"},
    log_level=LogLevel.INFO,
)

# Initialize the EvolDataGenerator
generator = EvolDataGenerator(config, output_file="output.jsonl")

# Generate the data
synthetic_data = generator.generate_data(contextual_tags, user_prompt)

# Print the result
print(synthetic_data)
