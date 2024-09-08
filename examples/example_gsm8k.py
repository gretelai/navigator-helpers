"""
This script generates synthetic data for GSM8K-like training examples.

It uses an LLM to create math problems similar to those found in the GSM8K dataset, 
focusing on grade school math word problems with step-by-step solutions.
"""

import textwrap
from typing import Dict, List

import numpy as np
import pandas as pd

from navigator_helpers import (
    DataFieldDefinition,
    DataModelDefinition,
    EvolDataGenerator,
    GeneratorConfig,
)


def create_contextual_tags(
    num_rows: int, *dataframes: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Creates contextual tags by sampling and combining data from provided dataframes.

    Args:
        num_rows (int): Number of rows to generate.
        dataframes (List[pd.DataFrame]): List of pandas dataframes from which to sample.

    Returns:
        pd.DataFrame: A dataframe containing the generated contextual tags.
    """
    total_combinations = np.prod([len(df.drop_duplicates()) for df in dataframes])
    print(
        f"Total number of unique combinations possible given contextual tags: {total_combinations}"
    )

    sampled_dfs = [
        df.sample(n=num_rows, replace=True).reset_index(drop=True) for df in dataframes
    ]
    df_contextual_tags = pd.concat(sampled_dfs, axis=1)
    df_contextual_tags.insert(0, "id", range(num_rows))

    print(f"Generated contextual tags:\n{df_contextual_tags.head()}")
    return df_contextual_tags


def get_gsm8k_evolutionary_strategies() -> Dict[str, List[str]]:
    """
    Returns a dictionary of evolutionary strategies to apply to the GSM8K-like dataset.

    Returns:
        Dict[str, List[str]]: Dictionary of strategies for improving questions and answers.
    """
    return {
        "improve_questions": [
            "Enhance the problem statement to improve diversity by introducing varied names, contexts, or scenarios, while ensuring it ends with a clear question and includes a question mark.",
            "Update the problem statement for improved coherence, making sure it logically flows and concludes with a specific question that prompts the solver to find a solution, including a question mark.",
            "Enhance the problem statement for improved readability, simplifying complex sentences and ensuring it ends with a well-formed question and a question mark.",
        ],
        "improve_answers": [
            "Rewrite the solution to the problem using clear step-by-step reasoning with one line per step, beginning with 'Let's solve this step by step:'. Arithmetic operations in each step must be annotated with <<calculation=result>>, and the final answer must be clearly separated and formatted as '#### (answer)'."
        ],
    }


GENERATION_PROMPT = textwrap.dedent(
    """
    You are tasked with generating diverse math problems similar to those in the GSM8K dataset. These problems should span from basic to advanced levels, covering various topics and requiring different levels of reasoning. Use diverse names, ethnicities, locations, and activities.

    Problems should range from simple one-step calculations to complex multi-step reasoning, incorporating topics like arithmetic, algebra, geometry, probability, rates, percentages, and optimization. Ensure diversity in contexts, names, and scenarios. For complex problems, include multiple constraints, variables, or conditions.

    Example Questions:
    1. John has 5 apples and buys 3 more. How many apples does he have now?
    2. A store offers a 20% discount, then takes an additional 10% off the discounted price. What's the total percentage discount?
    3. A company produces two types of products, A and B. The profit per unit of A is $30 and for B is $40. Each unit of A requires 2 hours of labor, while B requires 3 hours. The company has 100 hours of labor available. They can produce a maximum of 40 units of A due to material constraints. How many units of each product should the company produce to maximize profit?

    Example Answers:
    1. Let's solve this step by step:
       1. John starts with 5 apples.
       2. He buys 3 more apples.
       3. To find the total, we add the initial amount and the amount bought:
          <<5 + 3 = 8>>
       Therefore, John now has 8 apples.
       #### 8

    2. Let's solve this step by step:
       1. First, calculate the price after the 20% discount:
          100% - 20% = <<100 - 20 = 80>>% of the original price
       2. Then, apply the 10% discount to the new price:
          10% of 80% = <<10 * 80 / 100 = 8>>%
       3. The price after both discounts is:
          80% - 8% = <<80 - 8 = 72>>% of the original price
       4. The total discount is:
          100% - 72% = <<100 - 72 = 28>>%
       Therefore, the total percentage discount is 28%.
       #### 28

    3. Let's solve this step-by-step using linear programming concepts:
       1. Define variables: Let x be the number of units of A, and y be the number of units of B.
       2. Set up the objective function to maximize profit:
          Profit = 30x + 40y
       3. Constraints:
          Labor constraint: 2x + 3y ≤ 100
          Material constraint for A: x ≤ 40
          Non-negativity: x ≥ 0, y ≥ 0
       4. Solve graphically or using the corner point method:
          a) (x, y) = (0, 33.33) gives profit: <<30 * 0 + 40 * 33.33 = 1333.2>>
          b) (x, y) = (40, 6.67) gives profit: <<30 * 40 + 40 * 6.67 = 1466.8>>
          c) (x, y) = (35, 10) gives profit: <<30 * 35 + 40 * 10 = 1450>>
       5. The maximum profit occurs at (40, 6.67), but we need integer solutions.
       6. Checking the nearest integer points:
          (40, 6) gives profit: <<30 * 40 + 40 * 6 = 1440>>
          (39, 7) gives profit: <<30 * 39 + 40 * 7 = 1450>>
       Therefore, the company should produce 39 units of A and 7 units of B to maximize profit.
       #### 1450

    Remember to vary the complexity and ensure all problems are solvable with the information provided.
"""
)


def main():
    """
    Main function to generate synthetic GSM8K-like math problems.
    """
    print("Starting GSM8K-like data generation...")

    NUM_TAGS = 100

    # Define the dataframes for the contextual tags
    df_topics = pd.DataFrame(
        {
            "topic": [
                "arithmetic",
                "fractions",
                "percentages",
                "geometry",
                "basic algebra",
                "data interpretation",
                "word problems",
                "ratios and proportions",
            ],
            "topic_description": [
                "Basic operations like addition, subtraction, multiplication, and division",
                "Problems involving fractions, mixed numbers, and operations with fractions",
                "Calculations involving percentages, such as finding discounts or interest",
                "Problems related to shapes, areas, perimeters, and simple volumes",
                "Simple algebraic equations and reasoning appropriate for grade school",
                "Interpreting data from simple graphs, charts, and tables",
                "Word problems that translate everyday scenarios into mathematical problems",
                "Solving problems involving ratios, proportions, and comparisons",
            ],
        }
    )

    df_contexts = pd.DataFrame(
        {
            "context": [
                "shopping",
                "sports",
                "cooking",
                "travel",
                "school",
                "family",
                "outdoor activities",
                "hobbies and crafts",
                "holidays and celebrations",
                "animals and nature",
            ],
            "context_description": [
                "Scenarios involving purchases, discounts, allowances, and saving money",
                "Problems related to scores, team statistics, and basic sports math",
                "Recipe scaling, ingredient measurements, and cooking times",
                "Distance calculations, simple time zone changes, and travel planning",
                "Classroom scenarios, such as calculating grades, attendance, or supplies",
                "Family activities like sharing chores, planning events, or splitting costs",
                "Activities like camping, playing in the park, or simple outdoor games",
                "Problems related to personal hobbies like building models, drawing, or crafting",
                "Scenarios based on common holidays and family celebrations",
                "Basic problems related to animals, pets, and nature observations",
            ],
        }
    )

    df_difficulties = pd.DataFrame(
        {
            "difficulty": ["easy", "medium", "hard"],
            "difficulty_description": [
                "Simple one or two-step problems focusing on basic arithmetic or concepts",
                "Multi-step problems requiring moderate reasoning and understanding of concepts",
                "Complex multi-step problems that challenge students with more involved reasoning and multiple operations",
            ],
        }
    )

    df_student_age_group = pd.DataFrame(
        {
            "age_group": ["grades 2-3", "grades 4-5", "grade 6"],
            "age_group_description": [
                "Problems suitable for students aged 7-9 (grades 2-3)",
                "Problems suitable for students aged 9-11 (grades 4-5)",
                "Problems suitable for students aged 11-12 (grade 6)",
            ],
        }
    )

    # Set up generator configuration
    config = GeneratorConfig(
        api_key="prompt",
        llm_model="gretelai/gpt-auto",
        num_generations=1,
        log_level="INFO",
        use_reflection=True,
    )

    model_def = DataModelDefinition(
        generation_instructions=GENERATION_PROMPT,
        fields=[
            DataFieldDefinition(
                name="question",
                type="str",
                description="A math problem ranging from basic to advanced, incorporating various topics and real-world contexts. Must end with a question mark.",
                validator="A natural language math question that ends with a question mark",
                evolution_strategies=["improve_questions"],
                evolution_rate=0.0,
                store_full_reflection=False,
            ),
            DataFieldDefinition(
                name="answer",
                type="str",
                description="Detailed step-by-step solution with explanations, annotated arithmetic operations. Must end with '#### ' followed by the final numeric answer.",
                evolution_strategies=["improve_answers"],
                evolution_rate=0.0,
                store_full_reflection=False,
            ),
        ],
    )

    # Create contextual tags
    contextual_tags = create_contextual_tags(
        NUM_TAGS,
        df_difficulties,
        df_topics,
        df_contexts,
        df_student_age_group,
    )

    # Initialize and run the synthetic data generator
    generator = EvolDataGenerator(
        config,
        model_def,
        custom_evolutionary_strategies=get_gsm8k_evolutionary_strategies(),
    )

    synthetic_data = generator.generate_data(
        contextual_tags,
        output_file="gsm8k_synthetic_data.jsonl",
    )

    print("GSM8K-like problems generation complete.")


if __name__ == "__main__":
    main()
