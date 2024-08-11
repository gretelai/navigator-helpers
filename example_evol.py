import pandas as pd

from navigator_helpers import EvolDataGenerator

# Define the user prompt
user_prompt = """You are an expert in generating complex, nuanced, and context-rich questions along with comprehensive SQL-based answers. Your expertise spans various SQL concepts including advanced joins, subqueries, window functions, CTEs, and complex aggregations.

Generate a dataset with the following columns:
- `sql_prompt`: A complex and nuanced question related to SQL and databases
- `sql_context`: A single string containing multiple valid SQL table CREATE statements, separated by semicolons
- `sql`: A complete and executable SQL query to answer the prompt, with no markup or explanations
- `sql_explanation`: A step-by-step explanation of what the SQL query is doing"""

# Create or load contextual tags for diversity
contextual_tags = pd.read_csv("./docs/data/sql_contextual_tags.csv")

config = {
    "api_key": "prompt",
    "tabular_model": "gretelai/Llama-3.1-8B-Instruct",
    "llm_model": "gretelai/gpt-llama3-1-8b",
    "num_generations": 2,
    "population_size": 3,
    "expansion_size": 1,
    "mutation_rate": 0.6,
    "column_validators": {"sql_context": "sql:postgres", "sql": "sql:postgres"},
}

generator = EvolDataGenerator(config, output_file="output.jsonl")
result = generator.generate_data(contextual_tags, user_prompt)
print(result)
