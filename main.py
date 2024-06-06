import argparse
import logging
from datasets import load_dataset
import json
from data_augmentation import DataAugmentationConfig, DataAugmenter

def main(args):
    """
    Main function to run the data augmentation process.

    Args:
        args: Command-line arguments.
    """
    logging.basicConfig(level=args.loglevel)

    # Dataset configuration
    DATASET_NAME = "databricks/databricks-dolly-15k"
    MAX_WORDS = 400
    NUM_EXAMPLES = 2

    # Load and preprocess the dataset
    dataset = load_dataset(DATASET_NAME, split="train")
    df = (
        dataset.to_pandas()
        .map(
            lambda x: x.replace("\n", " ")
            .replace("\r", " ")
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        .assign(
            num_words=lambda df_: df_["context"]
            .str.cat(df_["response"], sep=" ")
            .str.split()
            .apply(len)
        )
        .query("num_words < @MAX_WORDS")
        .query("context != ''")
        .drop(columns=["category", "num_words"])
        .head(NUM_EXAMPLES)
        .reset_index(drop=True)
    )

    # Gretel API configuration
    GRETEL_API_KEY = "prompt"
    GRETEL_PRIMARY_MODEL = "gretelai/gpt-auto"
    MAX_CO_TEACH_LLMS = 3

    # Create the data augmentation configuration
    config = DataAugmentationConfig(
        num_instructions=5,
        num_responses=5,
        temperature=0.8,
        max_tokens_instruction=100,
        max_tokens_response=150,
        api_key=GRETEL_API_KEY,
        primary_model=GRETEL_PRIMARY_MODEL,
        max_co_teach_llms=MAX_CO_TEACH_LLMS,
    )
    config.add_field("context", field_type="context")
    config.add_field("instruction", field_type="instruction")
    config.add_field("response", field_type="response")

    # Create the data augmenter and perform augmentation
    augmenter = DataAugmenter(
        df,
        config,
        use_examples=True,
        use_aaa=args.use_aaa,
        output_file="results.csv",
    )
    new_df = augmenter.augment()

    # Print the augmented data as JSON
    print(json.dumps(new_df.to_dict(orient="records"), indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Data Augmentation with Evol-Instruct and Evol-Answer"
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--disable_aaa",
        dest="use_aaa",
        action="store_false",
        help="Disable AI Align AI (AAA) to improve runtime (default: False)",
        default=True,
    )
    args = parser.parse_args()
    args.loglevel = getattr(logging, args.loglevel)
    main(args)