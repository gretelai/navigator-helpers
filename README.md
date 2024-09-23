# Navigator Helpers 🚀

Gretel Navigator is a compound AI-based system for generating high-quality synthetic data using contextual tags and an evolutionary approach. This method iteratively improves, validates, and evaluates outputs to create synthetic data with greater quality than an underlying LLM could do on its own, and to combat hallucinations in AI-generated content.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Key Features](#key-features)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

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

### Configuration

The data synthesis configuration is defined in a YAML file. Here's a realistic example based on the `example_nl2sql.yml`:

```yaml
# Generator configuration
api_key: prompt
llm_model: gretelai/gpt-auto
log_level: INFO
use_reflection: true
output_prefix: basic_nl2sql
evol_generations: 1
num_examples: 1000

# Data model definition
generation_instructions: |
  You are a seasoned SQL expert specializing in crafting intricate, context-rich queries and explanations. 
  Use the provided contextual tags as instructions for generation. Reference the provided Context to ensure relevance and appropriate complexity.
  Use proper spacing for all generated SQL content, with no extra comments or explanations when generating code.
  All SQL data should be PostgreSQL compatible SQL.
  Avoid excessive newlines, especially at the end of generated content.

fields:
  - name: sql_context
    type: str
    description: A single string comprising multiple valid PostgreSQL CREATE TABLE statements and a complex schema similar to a production application including multiple tables, separated by semicolons. The schema should be based on the provided Context, particularly the domain and domain_description.
    validator: sql:postgres
    evolution_strategies:
      - Enhance the schema to include domain-specific tables and data types.
      - Add relevant indexes, constraints, and views reflecting real-world designs.
    evolution_rate: 0.1

  - name: prompt
    type: str
    description: A detailed, nuanced natural language prompt that a user might ask to a database for a particular task, based on the provided sql_context field that challenges advanced understanding. The prompt should align with the domain and domain_description from the contextual tags.
    evolution_strategies:
      - Refine the prompt to sound more natural.
      - Ensure the prompt reflects real-world business needs.
    evolution_rate: 0.1

  # ... (other fields omitted for brevity)

contextual_tags:
  tags:
    - name: sql_complexity
      values:
        - Moderate
        - Complex
        - Very Complex
        - Expert
      weights: [0.4, 0.3, 0.2, 0.1]
    
    - name: domain
      values:
        - Healthcare
        - Finance
        - E-commerce
        - Education
        - Manufacturing
      weights: [0.25, 0.25, 0.2, 0.15, 0.15]

    # ... (other tags omitted for brevity)
```

### Generating Synthetic Data

To generate synthetic data, use the `run_generation.py` script located in the `examples/` directory:

```bash
python examples/run_generation.py examples/example_nl2sql.yml
```

This script will:
1. Load the YAML configuration
2. Create a `DataModel` from the YAML
3. Initialize the `EvolDataGenerator`
4. Generate the data

The output will be saved to a file with the prefix specified in the YAML configuration (in this case, `basic_nl2sql`).

## Key Features

1. **YAML Configuration**: All configuration is now centralized in a YAML file, making it easier to manage and modify.

2. **Evolutionary Generations**: Specified by `evol_generations` in the YAML config.

3. **Contextual Tags**: Defined in the YAML config under `contextual_tags`.

4. **Built-in and Custom Evolutionary Strategies**: Defined for each field in the YAML config.

5. **Validators**: Specified for each field in the YAML config.

6. **Reflection**: Controlled by the `use_reflection` parameter in the YAML config.

7. **Flexible Output**: The `output_prefix` in the YAML config determines the output file name.

For more detailed information on these features, please refer to [ADVANCED_USAGE.md](ADVANCED_USAGE.md).

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue. For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the Gretel License. See the `LICENSE` file for details.