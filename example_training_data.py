import json
import logging
import pandas as pd
from navigator_helpers import DataSynthesisConfig, TrainingDataSynthesizer


def main():
    """
    Main function to run the training data synthesis process
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # AI Align AI (AAA) configuration
    USE_AAA = True  # Disable to improve runtime

    # Gretel API configuration
    GRETEL_API_KEY = "prompt"
    NAVIGATOR_TABULAR = "gretelai-azure/gpt-3.5-turbo" # "gretelai/auto"
    NAVIGATOR_LLM = "gretelai/gpt-auto"
    CO_TEACH_LLMS = [
        "gretelai/gpt-llama3-8b",
        # "gretelai/gpt-mistral7b",
    ]  # List of co-teaching models

    # Define a system prompt
    SYSTEM_PROMPT = """
    You are an AI assistant tasked with generating high-quality instruction-response pairs.
    Your goal is to create diverse, engaging, and informative content that covers a wide range of topics.
    When generating instructions, aim for clear, concise questions or commands that prompt thoughtful responses.
    When generating responses, provide detailed, accurate, and helpful information that directly addresses the instruction.
    """

    # Dataset configuration
    df = pd.read_csv(
        "https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/llm-training-data/databricks_dolly_instruction_set.csv",
        nrows=10,
    )
    print("Example record")
    print(json.dumps(df.head(1).to_dict(orient="records"), indent=2))

    # Create the training data synthesis configuration
    config = DataSynthesisConfig(
        input_fields=["context", "instruction", "response"],
        output_instruction_field="instruction",
        output_response_field="response",
        num_instructions=5,
        num_responses=5,
        temperature=0.8,
        max_tokens_instruction=100,
        max_tokens_response=150,
        api_key=GRETEL_API_KEY,
        navigator_tabular=NAVIGATOR_TABULAR,
        navigator_llm=NAVIGATOR_LLM,
        co_teach_llms=CO_TEACH_LLMS,
        system_prompt=SYSTEM_PROMPT,
        instruction_format_prompt="A well-formulated question or command in everyday English.",
        response_format_prompt="A well-formulated response to the question in everyday English.",
    )

    # Create the training data synthesizer and perform synthesis
    synthesizer = TrainingDataSynthesizer(
        df,
        config,
        use_aaa=USE_AAA,
        output_file="results.csv",
        verbose=True,
    )
    new_df = synthesizer.generate()

    # Print the augmented data as JSON
    print(json.dumps(new_df.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
