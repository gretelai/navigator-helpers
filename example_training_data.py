import json
import logging

import pandas as pd

from navigator_helpers import (InstructionResponseConfig,
                               TrainingDataSynthesizer)


def main():
    """
    Main function to run the training data synthesis process
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # Gretel API configuration
    GRETEL_API_KEY = "prompt"
    NAVIGATOR_TABULAR = "gretelai/auto"
    NAVIGATOR_LLM = "gretelai/gpt-auto"
    CO_TEACH_LLMS = [
        "gretelai/gpt-llama3-8b",
        "gretelai/gpt-mistral7b",
    ]

    # Define a system prompt
    SYSTEM_PROMPT = """You are an AI assistant tasked with generating high-quality instruction-response pairs.
Your goal is to create diverse, engaging, and informative content that covers a wide range of topics.
When generating instructions, aim for clear, concise questions or commands that prompt thoughtful responses.
When generating responses, provide detailed, accurate, and helpful information that directly addresses the instruction.
Focus on generating questions that can be answered directly from the given context, and provide responses that are concise and to the point.
"""

    # Dataset configuration
    df = pd.read_csv(
        "https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/llm-training-data/databricks_dolly_instruction_set.csv",
        nrows=10,
    )

    config = InstructionResponseConfig(
        input_fields=["context"],
        output_instruction_field="synthetic_instruction",
        output_response_field="synthetic_response",
        num_generations=3,
        population_size=5,
        mutation_rate=0.5,
        temperature=0.7,
        max_tokens=150,
        api_key=GRETEL_API_KEY,
        navigator_tabular=NAVIGATOR_TABULAR,
        navigator_llm=NAVIGATOR_LLM,
        co_teach_llms=CO_TEACH_LLMS,
        system_prompt=SYSTEM_PROMPT,
        instruction_format_prompt="Generate a concise and clear question or command directly related to the given context. The question should be specific, easy to understand, and focused on a single aspect of the context.",
        instruction_mutation_prompt="Modify this question to make it more focused and directly related to the main point of the context.",
        instruction_complexity_prompt="Rate the complexity of this question based on its use of technical jargon and depth of explanation:",
        instruction_quality_prompt="Evaluate the quality of this question based on its clarity, relevance, and completeness:",
        response_format_prompt="Generate a concise and direct response to the given question. The response should be clear, factually accurate, and focused solely on answering the question without additional context or information.",
        response_mutation_prompt="Modify this response to make it more concise and directly address the question asked.",
        response_complexity_prompt="Rate the complexity of this response based on its use of technical jargon and depth of explanation:",
        response_quality_prompt="Evaluate the quality of this response based on its clarity, relevance, and completeness:",
        instruction_complexity_target=0.2,  # For simpler instructions
        response_complexity_target=0.2,  # For simpler responses
        use_aaa=True,
    )

    with open("default.json", "w") as f:
        f.write(json.dumps(config.to_dict()))

    # Create the training data synthesizer and perform synthesis
    synthesizer = TrainingDataSynthesizer(
        df,
        config,
        output_file="results.jsonl",
        verbose=True,
    )
    new_df = synthesizer.generate()

    # Print the first few rows of the synthetic data
    print(new_df.head().to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
