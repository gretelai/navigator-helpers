import logging
import textwrap

from navigator_helpers.config import ConfigGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_FILE = "synthetic_config.yaml"
DEFAULT_MODEL = "gretelai/gpt-auto"


def get_diversity_target():
    default_diversity = 10000
    prompt = textwrap.dedent(
        f"""
        Enter the desired diversity target (press Enter to use default: {default_diversity}):
        - 10000 is a normal diversity target
        - 100000 is considered high diversity
        Your choice: """
    )

    
def get_user_task():
    default_task = "Create a dataset of high-quality math problems with solutions, similar to GSM8K."
    user_task = input(
        f"\nPlease describe the dataset you want to generate (press Enter to use default: '{default_task}'): "
    ).strip()
    return user_task if user_task else default_task


def main():
    # Initialize ConfigGenerator
    config_generator = ConfigGenerator(api_key="prompt", model=DEFAULT_MODEL)

    # Get user task
    user_task = get_user_task()
    logger.info(f"User task: {user_task}")

    # Set user task
    config_generator.set_user_task(user_task)

    # Prompt user for diversity target or use default
    diversity_target = get_diversity_target()
    logger.info(f"diversity target: {diversity_target}")

    # Generate tags with the specified diversity target
    logger.info("Generating tags...")
    tags = config_generator.generate_tags(diversity_target=diversity_target)
    logger.info("Generated tags:")
    logger.info(tags.to_yaml())

    # Generate data model
    logger.info("Generating data model...")
    data_model = config_generator.generate_data_model()
    logger.info("Generated data model")

    # Get config
    config = config_generator.get_config()
    logger.info(f"Synthetic Data Generation Configuration:\n'''\n{config}\n'''")

    # Save the final config to a file
    with open(OUTPUT_FILE, "w") as f:
        f.write(config)
    logger.info(f"Final configuration saved to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()
