"""
This script generates synthetic data for GSM8K-like training examples.

It uses an LLM to create math problems similar to those found in the GSM8K dataset,
focusing on grade school math word problems with step-by-step solutions.
"""

import textwrap

from typing import Dict, List

import pandas as pd

from navigator_helpers import (
    batch_and_write_data,
    DataFieldDefinition,
    DataModelDefinition,
    EvolDataGenerator,
    GeneratorConfig,
    mix_contextual_tags,
)

# Constants
NUM_TAGS = 100000
BATCH_SIZE = 20


def get_contextual_dataframes() -> List[pd.DataFrame]:
    """
    Returns a list of dataframes used for contextual tags.
    """
    df_topics = pd.DataFrame(
        {
            "topic": [
                "arithmetic",
                "fractions",
                "percentages",
                "geometry",
                "algebra",
                "data interpretation",
                "ratios",
                "proportions",
                "compound interest",
                "polynomials",
                "exponential growth/decay",
                "probability",
                "optimization",
            ]
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
            ]
        }
    )

    df_difficulties = pd.DataFrame(
        {
            "difficulty": ["easy", "medium", "hard", "very hard"],
            "difficulty_description": [
                "Simple one or two-step problems focusing on basic arithmetic or concepts",
                "Multi-step problems requiring moderate reasoning and understanding of concepts",
                "Complex multi-step problems with multiple variables and operations",
                "Advanced problems requiring systems of equations, conditional logic, and optimization.",
            ],
        }
    )

    df_student_age_group = pd.DataFrame(
        {
            "age_group": ["grade 6-7", "grades 8-9", "grades 10-12"],
        }
    )

    diverse_cultures = pd.DataFrame(
        {
            "culture": [
                "American",
                "Mexican",
                "Canadian",
                "European",
                "Asian",
                "African",
            ]
        }
    )

    return [
        df_difficulties,
        df_topics,
        df_contexts,
        df_student_age_group,
        diverse_cultures,
    ]


def create_model_definition() -> DataModelDefinition:
    """
    Creates and returns the DataModelDefinition for GSM8K-like problems.
    """
    generation_instructions = textwrap.dedent(
        """
        You are tasked with generating diverse math problems similar to those in the GSM8K dataset. These problems should span from basic to advanced levels, covering various topics and requiring different levels of reasoning. Use diverse names, ethnicities, locations, and activities.

        Problems should focus on complex multi-step reasoning, incorporating topics like arithmetic, algebra, geometry, probability, rates, percentages, and optimization. Ensure diversity in contexts, names, and scenarios. For complex problems, include multiple constraints, variables, or conditions.

        Example Questions:
        1. A store offers a 20% discount, then takes an additional 10% off the discounted price. What's the total percentage discount?
        2. A company produces two types of products, A and B. The profit per unit of A is $30 and for B is $40. Each unit of A requires 2 hours of labor, while B requires 3 hours. The company has 100 hours of labor available. They can produce a maximum of 40 units of A due to material constraints. How many units of each product should the company produce to maximize profit?

        Example Answers:

        1. Let's solve this step by step:
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

        2. Let's solve this step-by-step using linear programming concepts:
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

    return DataModelDefinition(
        generation_instructions=generation_instructions,
        fields=[
            DataFieldDefinition(
                name="question",
                type="str",
                description="A math problem ranging from basic to advanced, incorporating various topics and real-world contexts. This is a problem for a student, do not include the answer. Must end with a question mark.",
                validator="A natural language math problem that ends with a question mark",
                evolution_strategies=[
                    "Increase the complexity by adding more steps to the problem.",
                    "Introduce diverse contexts and new mathematical concepts.",
                    "Enhance the wording for better clarity and engagement.",
                ],
                evolution_rate=0.1,
            ),
            DataFieldDefinition(
                name="answer",
                type="str",
                description="Detailed step-by-step solution with explanations. EVERY arithmetic or algebraic operation must be annotated with <<calculation=result>>. Must end with '#### ' followed by the final numeric answer.",
                evolution_strategies=[
                    "Refine the solution to improve step-by-step clarity.",
                    "Add detailed annotations for all key calculations.",
                    "Simplify the explanation while maintaining accuracy and thoroughness.",
                ],
                evolution_rate=0.1,
                store_full_reflection=True,
            ),
        ],
    )


def main():
    """
    Main function to generate synthetic GSM8K-like math problems.
    """
    print("Starting advanced GSM8K-like data generation...")

    config = GeneratorConfig(
        api_key="prompt",
        llm_model="gretelai/gpt-auto",
        num_generations=1,
        log_level="INFO",
        use_reflection=True,
    )

    contextual_tags = mix_contextual_tags(NUM_TAGS, get_contextual_dataframes())

    generator = EvolDataGenerator(
        config,
        create_model_definition(),
    )

    batch_and_write_data(
        generator=generator,
        contextual_tags=contextual_tags,
        batch_size=BATCH_SIZE,
        file_prefix="advanced_gsm8k",
    )

    print("Advanced GSM8K scenario generation complete.")


if __name__ == "__main__":
    main()
