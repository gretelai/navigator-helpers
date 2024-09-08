"""
This script generates synthetic data for closed question-answer pairs based on snippets from Wikipedia.

It uses an LLM to create context-specific questions and concise, informative answers 
based solely on the provided context.
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
    config = GeneratorConfig(
        api_key="prompt",
        llm_model="gretelai/gpt-auto",
        num_generations=1,
        log_level="INFO",
        use_reflection=True,
    )

    model_def = DataModelDefinition(
        generation_instructions=textwrap.dedent(
            """You are an expert in generating balanced, context-rich questions and comprehensive answers based on given contexts. Your task is to:
            1. Create questions that are specific, clear, and directly related to key points in the given context.
            2. Ensure questions include sufficient background information for understanding without prior knowledge.
            3. Craft questions that can be answered using only the information provided in the context.
            4. Generate answers that are informative, concise when possible, and detailed when necessary.
            5. Provide comprehensive responses that fully address the question, including relevant context and explanations.
            6. Maintain consistency between the question, answer, and the original context.
            7. Avoid revealing the answer in the question.
            8. Adapt the level of detail in the answer based on the complexity of the topic.
            Remember, the goal is to create question-answer pairs that are informative, understandable, and faithful to the provided context."""
        ),
        fields=[
            DataFieldDefinition(
                name="question",
                type="str",
                description="""Generate a specific and clear question directly related to a key point in the given context. The question should:
                1. Include enough background information to be understood without prior knowledge.
                2. Be answerable using only the information provided in the context.
                3. Not reveal the answer.
                4. Be focused and allow for a concise answer if the information permits, or a more detailed response when necessary.
                5. Encourage critical thinking or analysis of the information in the context.""",
                validator="Question",
                evolution_strategies=["complexity", "improve", "diversity", "simplify"],
                evolution_rate=0.2,
            ),
            DataFieldDefinition(
                name="response",
                type="str",
                description="""Generate an informative answer to the given question. The response should:
                1. Use only the information provided in the original context.
                2. Be as concise as possible while fully addressing the question.
                3. Include relevant context and explanations where necessary.
                4. Provide a more detailed response for complex topics.
                5. Ensure enough background information is given to be understood by someone unfamiliar with the topic.
                6. Maintain consistency with the question and the original context.
                7. Avoid introducing information not present in the original context.""",
                validator="Answer",
                evolution_strategies=["complexity", "improve", "simplify"],
                evolution_rate=0.2,
            ),
        ],
    )

    # Load Wikipedia snippets from Databricks Dolly as contextual tags
    contextual_tags = pd.read_csv(
        "https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/llm-training-data/databricks_dolly_instruction_set.csv",
        nrows=10,
        usecols=["context"],
    )

    # Initialize the SyntheticDataGenerator
    generator = EvolDataGenerator(config, model_def)

    # Generate the data
    synthetic_data = generator.generate_data(
        contextual_tags, output_file="closed_qa_synthetic_data.jsonl"
    )
    print("Closed Question/Ansewr data generation complete.")


if __name__ == "__main__":
    main()
