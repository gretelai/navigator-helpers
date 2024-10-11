import argparse

from navigator_helpers.data_models import DataModel
from navigator_helpers.synthetic_data_generator import SyntheticDataGenerator


def run_generation(config_file: str):
    # Load the YAML configuration
    with open(config_file, "r") as file:
        yaml_content = file.read()

    # Create the DataModelDefinition (all configuration in one place)
    model_definition = DataModel.from_yaml(yaml_content)

    # Initialize the SyntheticDataGenerator
    generator = SyntheticDataGenerator(model_definition)

    # Generate the data and process each record as it's yielded
    for record in generator.generate_data():
        pass

    latest_filename = generator.output_filename
    print(f"Data generation complete. Output saved to {latest_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic data using a YAML config."
    )
    parser.add_argument("config", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()
    run_generation(args.config)
