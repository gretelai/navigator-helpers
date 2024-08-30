import textwrap
from typing import Dict, List
import pandas as pd
from navigator_helpers import (
    DataFieldDefinition,
    DataModelDefinition,
    EvolDataGenerator,
    GeneratorConfig,
)


def create_contextual_tags(num_rows, *dataframes):
    sampled_dfs = [
        df.sample(n=num_rows, replace=True).reset_index(drop=True) for df in dataframes
    ]
    df_contextual_tags = pd.concat(sampled_dfs, axis=1)
    df_contextual_tags.insert(0, "id", range(num_rows))
    return df_contextual_tags


def get_gsm8k_evolutionary_strategies() -> Dict[str, List[str]]:
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


def main():
    NUM_TAGS = 100

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

    df_steps = pd.DataFrame(
        {
            "steps": list(
                range(2, 9)
            ),  # Two to eight steps to reflect typical GSM8K problems
            "steps_description": [
                "Two-step problem",
                "Three-step problem",
                "Four-step problem",
                "Five-step problem",
                "Six-step problem",
                "Seven-step problem",
                "Eight-step problem",
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

    config = GeneratorConfig(
        api_key="prompt",
        llm_model="gretelai/gpt-auto",
        num_generations=1,
        log_level="INFO",
    )

    model_def = DataModelDefinition(
        system_message=textwrap.dedent(
            """Create grade school math word problems with step-by-step solutions using the provided tags for difficulty, topic, context, and number of steps. Each solution should primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+, −, *, /) to reach the final answer.

            IMPORTANT:
            - Annotate all steps containing arithmetic operations in the format <<calculation=result>>.
            - The final answer must be clearly separated and preceded by '#### ' (a space character follows '####').
            - Ensure diversity in names, ethnicities, genders, and activities by using a wide range of names and scenarios.
            - Provide clear, logical step-by-step reasoning that leads to the final answer.

            Example:

            Question: Anjali sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Anjali sell altogether in April and May?

            Answer: Let's solve this step by step:
            1. First, we need to calculate how many clips Anjali sold in May.
            She sold half as many as in April, so we divide 48 by 2.
            Clips sold in May = <<48/2=24>>24

            2. Now we know Anjali sold 48 clips in April and 24 clips in May.
            To find the total, we add these numbers together.
            Total clips sold = <<48+24=72>>72

            Therefore, Anjali sold 72 clips altogether in April and May.

            #### 72

            Failure to follow this format will result in rejection."""
        ),
        fields=[
            DataFieldDefinition(
                name="question",
                type="str",
                description="A grade school math word problem based on the given contextual tags, which contains a detailed problem WITHOUT THE ANSWER.",
                validator="A natural language math problem written in English",
                evolution_strategies=["improve_questions"],
                evolution_rate=1.0,
            ),
            DataFieldDefinition(
                name="answer",
                type="str",
                description="Step-by-step solution with calculator annotations for EVERY step that involves mathematical operations. MUST end with '####' followed by a space character and the final numeric answer.",
                evolution_strategies=["improve_answers"],
                evolution_rate=1.0,
            ),
        ],
    )

    contextual_tags = create_contextual_tags(
        NUM_TAGS,
        df_difficulties,
        df_topics,
        df_contexts,
        df_steps,
        df_student_age_group,
    )

    print(contextual_tags)

    generator = EvolDataGenerator(
        config,
        model_def,
        custom_evolutionary_strategies=get_gsm8k_evolutionary_strategies(),
    )

    synthetic_data = generator.generate_data(
        contextual_tags,
        output_file="gsm8k_synthetic_data.jsonl",
    )
    print("GSM8k-like problems generation complete.")


if __name__ == "__main__":
    main()
