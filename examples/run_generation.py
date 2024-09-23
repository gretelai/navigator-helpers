import argparse

from navigator_helpers.data_models import DataModel
from navigator_helpers.evolutionary_data_generator import EvolDataGenerator


def run_generation(config_file: str):
    # Load the YAML configuration
    with open(config_file, "r") as file:
        yaml_content = file.read()

    # Create the DataModelDefinition (all configuration in one place)
    model_definition = DataModel.from_yaml(yaml_content)

    # Initialize the EvolDataGenerator
    generator = EvolDataGenerator(model_definition)

    # Generate the data
    generator.generate_data()

    print(f"Data generation complete. Output saved to {model_definition.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using a YAML config."
    )
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()
    run_generation(args.config)
