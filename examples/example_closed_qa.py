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
        num_generations=3,
        log_level="INFO",
    )

    model_def = DataModelDefinition(
        system_message=textwrap.dedent(
            """"You are an expert in generating balanced, context-rich questions and comprehensive answers based on given contexts. Your goal is to create question-answer pairs that are informative, detailed when necessary, and understandable without prior knowledge, while not revealing the answer in the question."""
        ),
        fields=[
            DataFieldDefinition(
                name="question",
                type="str",
                description="""Generate a specific and clear question directly related to a key point in the given context. The question should include enough background information to be understood without prior knowledge, while being answerable using only the information provided. Do not reveal the answer in the question. Ensure the question is focused and can be answered concisely if the information allows, but also accommodate for more detailed responses when appropriate.""",
                validator="Question",
                evolution_strategies=["complexity", "improve", "diversity", "simplify"],
            ),
            DataFieldDefinition(
                name="response",
                type="str",
                description="""Generate an informative answer to the given question. Use only the information provided in the original context. The response should be as concise as possible while fully addressing the question, including relevant context and explanations where necessary. For complex topics, provide a more detailed response. Ensure the answer provides enough background information to be understood by someone unfamiliar with the topic.""",
                validator="Answer",
                evolution_strategies=["complexity", "improve", "simplify"],
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
    print("Synthetic data generation complete.")


if __name__ == "__main__":
    main()
