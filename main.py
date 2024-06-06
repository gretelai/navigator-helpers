import json
import pandas as pd
from data_augmentation import DataAugmentationConfig, DataAugmenter

def main():
    """
    Main function to run the data augmentation process.
    """
    # AI Align AI (AAA) configuration
    USE_AAA = True # Disable to improve runtime

    # Gretel API configuration
    GRETEL_API_KEY = "prompt"
    GRETEL_PRIMARY_MODEL = "gretelai/gpt-llama3-8b"
    CO_TEACH_MODELS = [
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
        num_instructions=5,
        num_responses=5,
        temperature=0.8,
        max_tokens_instruction=100,
        max_tokens_response=150,
        api_key=GRETEL_API_KEY,
        primary_model=GRETEL_PRIMARY_MODEL,
        co_teach_models=CO_TEACH_MODELS,
        instruction_format_prompt="A well-formulated question or command in everyday English.",
        response_format_prompt="A well-formulated response to the question in everyday English.",
    )
    config.add_field("context", field_type="context")
    config.add_field("instruction", field_type="instruction")
    config.add_field("response", field_type="response")

    # Create the data augmenter and perform augmentation
    augmenter = DataAugmenter(
        df,
        config,
        use_examples=True,
        use_aaa=USE_AAA,
        output_file="results.csv",
    )
    new_df = augmenter.augment()

    # Print the augmented data as JSON
    print(json.dumps(new_df.to_dict(orient="records"), indent=2))

if __name__ == "__main__":
    main()