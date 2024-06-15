import json
import logging
import pandas as pd
from navigator_helpers import DataAugmentationConfig, DataAugmenter


def main():
    """
    Main function to run the data augmentation process.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # AI Align AI (AAA) configuration
    USE_AAA = True  # Disable to improve runtime

    # Gretel API configuration
    GRETEL_API_KEY = "prompt"
    NAVIGATOR_TABULAR = "gretelai/auto"
    NAVIGATOR_LLM = "gretelai/gpt-auto"
    CO_TEACH_LLMS = [
        "gretelai/gpt-llama3-8b",
        "gretelai/gpt-mistral7b",
    ]  # List of co-teaching models

    # Dataset configuration
    df = pd.read_csv(
        "https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/llm-training-data/databricks_dolly_instruction_set.csv",
        nrows=1,
    )
    print("Example record")
    print(json.dumps(df.head(1).to_dict(orient="records"), indent=2))

    # Create the data augmentation configuration
    config = DataAugmentationConfig(
        input_fields=["context", "instruction", "response"],
        output_instruction_field="gen_instruction",
        output_response_field="gen_response",
        num_instructions=5,
        num_responses=5,
        temperature=0.8,
        max_tokens_instruction=100,
        max_tokens_response=150,
        api_key=GRETEL_API_KEY,
        navigator_tabular=NAVIGATOR_TABULAR,
        navigator_llm=NAVIGATOR_LLM,
        co_teach_llms=CO_TEACH_LLMS,
        instruction_format_prompt="A well-formulated question or command in everyday English.",
        response_format_prompt="A well-formulated response to the question in everyday English."
    )

    # Create the data augmenter and perform augmentation
    augmenter = DataAugmenter(
        df,
        config,
        use_aaa=USE_AAA,
        output_file="results.csv",
        verbose=True,
    )
    new_df = augmenter.augment()

    # Print the augmented data as JSON
    print(json.dumps(new_df.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
