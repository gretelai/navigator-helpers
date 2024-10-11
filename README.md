# Navigator Helpers 🚀

Gretel Navigator is a compound AI-based system for generating high-quality synthetic data using contextual tags and an evolutionary approach. This method iteratively improves, validates, and evaluates outputs to create synthetic data with greater quality than an underlying LLM could do on its own, and to combat hallucinations in AI-generated content.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Key Features](#key-features)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gretelai/navigator-helpers.git
   cd navigator-helpers
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Basic Usage

### Configuration

The data synthesis configuration is defined in a YAML file. Here's a simplified example:

```yaml
# Generator configuration
api_key: prompt
llm_model: gretelai/gpt-auto
log_level: INFO
use_reflection: true
output_filename: synthetic_data.jsonl
evolution_generations: 1
num_examples: 1000

# Data model definition
generation_instructions: |
  You are a seasoned SQL expert specializing in crafting intricate, context-rich queries and explanations. 
  Use the provided contextual tags as instructions for generation.

fields:
  - name: sql_context
    type: str
    description: A single string comprising multiple valid PostgreSQL CREATE TABLE statements.
    validator: sql:postgres

  - name: prompt
    type: str
    description: A detailed, nuanced natural language prompt that a user might ask to a database for a particular task.

contextual_tags:
  tags:
    - name: sql_complexity
      values:
        - value: Moderate
          weight: 0.4
        - value: Complex
          weight: 0.3
        - value: Very Complex
          weight: 0.2
        - value: Expert
          weight: 0.1

evolution:
  rate: 0.1
  strategies:
    - Enhance the schema to include domain-specific tables and data types.
    - Add relevant indexes, constraints, and views reflecting real-world designs.
```

### Generating Synthetic Data

To generate synthetic data, use the `run_generation.py` script:

```bash
python examples/run_generation.py examples/example_nl2sql.yml
```

This script will:
1. Load the YAML configuration
2. Create a `DataModel` from the YAML
3. Initialize the `EvolDataGenerator`
4. Generate the data

The output will be saved to the file specified by `output_filename` in the YAML configuration.

## Key Features

1. **YAML Configuration**: All configuration is centralized in a YAML file, making it easier to manage and modify.

2. **Evolutionary Process**: Controlled by the `evolution` section in the YAML config, including strategies and rate.

3. **Contextual Tags**: Defined in the YAML config under `contextual_tags`, supporting weighted values.

4. **Field-specific Configuration**: Each field can have its own type, description, and validator.

5. **Flexible Validation**: Validators can be specified for each field in the YAML config.

6. **Reflection**: Controlled by the `use_reflection` parameter in the YAML config.

7. **Customizable Output**: The `output_filename` in the YAML config determines the output file name.

For more detailed information on these features, please refer to the source code and comments within the project.

## Examples

Example YAML configurations and usage scenarios can be found in the `examples/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the Gretel License. See the `LICENSE` file for details.