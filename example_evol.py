import pandas as pd

from navigator_helpers import (
    EvolDataGenerator,
    GeneratorConfig,
    MutationCategory,
    LogLevel,
)

# Define the user prompt
user_prompt = """You are an expert in generating complex, nuanced, and context-rich questions along with comprehensive SQL-based answers. Your expertise spans various SQL concepts including advanced joins, subqueries, window functions, CTEs, and complex aggregations.
Generate a dataset with the following columns:
- `sql_prompt`: A complex and nuanced question related to SQL and databases
- `sql_context`: A single string containing multiple valid SQL table CREATE statements, separated by semicolons
- `sql`: A complete and executable SQL query to answer the prompt, with no markup or explanations
- `sql_explanation`: A step-by-step explanation of what the SQL query is doing"""

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
