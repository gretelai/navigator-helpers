import logging

from gretel_client import Gretel

from navigator_helpers.config import ConfigGenerator
from navigator_helpers.text_inference import TextInference

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_FILE = "final_config.yaml"
MODEL = "gretelai/gpt-auto"


def main():
    # Initialize Gretel
    gretel = Gretel(api_key="prompt")
    llm = gretel.factories.initialize_navigator_api(
        "natural_language", backend_model=MODEL
    )
    text_inference = TextInference(llm, logging.getLogger(__name__))

    # Initialize ConfigGenerator
    config_generator = ConfigGenerator(text_inference)

    # Prompt user for task or use default
    default_task = "Create a dataset of high-quality math problems with solutions, similar to GSM8K."
    user_task = input(
        f"\nPlease describe the dataset you want to generate (press Enter to use default: '{default_task}'): "
    ).strip()
    print()

    if not user_task:
        user_task = default_task

    logger.info(f"User task: {user_task}")

    # Set user task
    config_generator.set_user_task(user_task)

    # Generate tags
    logger.info("Generating tags...")
    tags = config_generator.generate_tags()
    logger.info("Generated tags:")
    logger.info(tags.to_yaml())

    # Generate data model
    logger.info("Generating data model...")
    data_model = config_generator.generate_data_model()
    logger.info("Generated data model:")
    logger.info(data_model.to_yaml())

    # Get final config
    final_config = config_generator.get_final_config()
    logger.info("Final configuration:")
    logger.info(final_config)

    # Save the final config to a file
    with open(OUTPUT_FILE, "w") as f:
        f.write(final_config)
    logger.info(f"Final configuration saved to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
