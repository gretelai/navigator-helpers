import logging

import pandas as pd

from datasets import load_dataset

from navigator_helpers import InstructionResponseConfig, TrainingDataSynthesizer

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(message)s")

API_KEY = "prompt"

# Load the dataset from Hugging Face
dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")

# Convert the first 100 rows to a pandas DataFrame
df = pd.DataFrame(dataset[:100])

# Display the first few rows of the DataFrame to verify
print(df.head())

# Define the prompts
system_prompt = """You are an expert in generating complex, nuanced, and context-rich questions along with comprehensive SQL-based answers. Your expertise spans various SQL concepts including advanced joins, subqueries, window functions, CTEs, and complex aggregations. Your goal is to create challenging question-answer pairs that require deep understanding of SQL and database concepts, while ensuring the questions are clear and the answers are syntactically correct SQL statements. You return only questions or answers and no extra explanations or comments."""

instruction_format_prompt = """Generate a complex and nuanced question related to the given SQL context and complexity description. The question should:
1. Incorporate advanced SQL concepts mentioned in the sql_complexity_description.
2. Require multiple steps or nested queries to solve.
3. Integrate business logic relevant to the domain and task type.
4. Challenge the respondent to think critically about query optimization and performance.
5. Include enough context to be understood without prior knowledge of the specific database.
6. Not reveal the exact SQL solution in the question itself.
Ensure the question is clear and focused, allowing for a comprehensive SQL-based answer.
Respond directly without explanations or comments."""

instruction_mutation_prompt = """Enhance the complexity and nuance of this question by:
1. Introducing additional SQL concepts or combining multiple concepts.
2. Adding constraints or conditions that require more sophisticated query logic.
3. Incorporating scenario-based elements that reflect real-world data challenges.
4. Encouraging the use of advanced SQL features like window functions, CTEs, or pivot operations where appropriate.
5. Ensuring the question remains clear and answerable using only the provided context.
Respond directly without explanations or comments."""

instruction_quality_prompt = """Evaluate the quality and complexity of this question based on:
1. Incorporation of advanced SQL concepts from the sql_complexity_description.
2. Requirement for multi-step or nested query solutions.
3. Integration of relevant business logic and domain knowledge.
4. Potential for query optimization considerations.
5. Clarity and specificity without revealing the full solution.
6. Appropriateness for the given sql_task_type.
7. Overall challenge level for an SQL expert."""

response_format_prompt = """Generate a comprehensive SQL answer to the given question. Your response should:
1. Provide a complete, syntactically correct SQL statement that fully addresses the question.
2. Utilize advanced SQL features and concepts appropriate to the complexity level indicated.
3. Include comments explaining the logic behind complex parts of the query.
4. Consider query performance and optimization where relevant.
5. Use appropriate table aliases and meaningful column names.
6. Incorporate error handling or null value management if necessary.
7. Follow SQL best practices and conventions.
Ensure the SQL statement is executable and would produce the correct result based on the given context.
Respond directly without explanations or comments."""

response_mutation_prompt = """Refine this SQL answer to enhance its complexity and effectiveness:
1. Introduce more advanced SQL features if applicable (e.g., window functions, CTEs, subqueries).
2. Optimize the query structure for better performance, if possible.
3. Add or modify comments to explain intricate parts of the query.
4. Ensure all necessary joins, filters, and aggregations are present and correct.
5. Verify that the query handles potential edge cases or null values.
6. Confirm that the SQL syntax is valid and follows best practices.
Respond directly without explanations or comments."""

response_quality_prompt = """Evaluate the quality of this SQL answer based on:
1. Correctness and completeness in addressing the question.
2. Appropriate use of advanced SQL features and concepts.
3. Query structure and potential for optimization.
4. Clarity of logic and helpful comments.
5. Handling of potential edge cases or data inconsistencies.
6. Adherence to SQL best practices and conventions.
7. Overall sophistication and effectiveness of the solution."""

# Define tabular and LLM configurations
NAVIGATOR_TABULAR = "gretelai/auto"
NAVIGATOR_LLM = "gretelai/gpt-auto"
CO_TEACH_LLMS = [
    "gretelai/gpt-llama3-1-8b",
    "gretelai/gpt-mistral-nemo-2407",
]  # List of co-teaching models


# Create the instruction response configuration
config = InstructionResponseConfig(
    input_fields=[
        "id",
        "domain",
        "domain_description",
        "sql_complexity",
        "sql_complexity_description",
        "sql_task_type",
        "sql_task_type_description",
        "sql_prompt",
        "sql_context",
        "sql",
        "sql_explanation",
    ],
    output_instruction_field="sql_prompt",
    output_response_field="sql",
    num_generations=2,
    population_size=5,
    mutation_rate=0.6,
    temperature=0.8,
    max_tokens=300,
    api_key=API_KEY,
    endpoint="https://api.gretel.ai",
    navigator_tabular=NAVIGATOR_TABULAR,
    navigator_llm=NAVIGATOR_LLM,
    co_teach_llms=CO_TEACH_LLMS,
    system_prompt=system_prompt,
    instruction_format_prompt=instruction_format_prompt,
    instruction_mutation_prompt=instruction_mutation_prompt,
    instruction_quality_prompt=instruction_quality_prompt,
    instruction_complexity_target=3,
    response_format_prompt=response_format_prompt,
    response_mutation_prompt=response_mutation_prompt,
    response_quality_prompt=response_quality_prompt,
    response_complexity_target=5,
    use_aaa=True,
)

# Create the training data synthesizer and perform synthesis
synthesizer = TrainingDataSynthesizer(
    df,
    config,
    output_file="data/sql-results.jsonl",
    verbose=True,
)
new_df = synthesizer.generate()
