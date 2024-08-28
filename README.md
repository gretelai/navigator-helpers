# Navigator Helpers 🚀

Gretel Navigator is a compound AI-based system for generating high-quality synthetic data using contextual tags and an evolutionary approach. This method iteratively improves, validates, and evaluates outputs to create synthetic data with greater quality than an underlying LLM could do on its own, and to combat hallucinations in AI-generated content.

## Table of Contents

- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

### Standard Installation

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

### Development Setup

For developers, additional setup is required to facilitate ongoing development:

1. Install the package in editable mode:
   ```bash
   make pip_install_dev
   ```

2. Common development tasks are available as `make` commands:
   - Apply consistent formatting:
     ```bash
     make style
     ```
   - Run tests:
     ```bash
     make test
     ```

## Basic Usage

### Input Requirements

The input to this program is LLM training data in a pandas DataFrame format (optional), containing tags that can be used to seed LLM generation for the specified task. You must specify one or more output fields to synthetically generate.

### Configuration

The data synthesis configuration is created using the `GeneratorConfig` class:

```python
config = GeneratorConfig(
    api_key="your_api_key",
    llm_model="gretelai/gpt-auto",
    num_generations=3,
    evolution_rate=0.5,
    log_level="INFO",
)
```

### Defining Your Data Model

Use `DataModelDefinition`, which utilizes Pydantic, to specify the structure of your synthetic data schema:

```python
model_def = DataModelDefinition(
    system_message="Your system message here",
    fields=[
        DataFieldDefinition(
            name="field_name",
            type="str",
            description="Field description",
            validator="your_validator",
            evolutionary_strategies=["strategy1", "strategy2"],
        ),
        # Add more fields as needed
    ],
)
```

### Generating Synthetic Data

To run the evolutionary data generation process:

```python
generator = EvolDataGenerator(config, model_def)
evolved_data = generator.generate_data(
    contextual_tags=your_contextual_data,
    output_file="output.jsonl",
    custom_evolutionary_strategies=your_custom_strategies  # Optional
)
```

For more detailed usage, including custom evolutionary strategies and validators, see [ADVANCED_USAGE.md](ADVANCED_USAGE.md).

## Examples

For detailed examples, please refer to the `examples/` directory in this repository. You can run these examples using:

```bash
python examples/example_nl2sql.py
```

or

```bash
python examples/example_closed_qa.py
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue. For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the Gretel License. See the `LICENSE` file for details.

## Acknowledgements

This project was inspired by the techniques described in the following papers and blog posts:

- Shazeer et al. "WizardLM-2: Empowering Large Language Models with Diverse Knowledge and Progressive Learning." WizardLM 2, 15 Apr. 2024, https://wizardlm.github.io/WizardLM2/ and https://arxiv.org/pdf/2304.12244.
- Yuan et al. "Self-Rewarding Language Models." arXiv preprint arXiv:2401.10020 (2024).
- Wei et al. "StarCoder2-Instruct: Fully Transparent and Permissive Self-Alignment for Code Generation." Hugging Face Blog, 29 Apr. 2024, https://huggingface.co/blog/sc2-instruct.
- Li et al. "Textbooks Are All You Need II: phi-1.5 technical report." arXiv preprint arXiv:2309.05463 (2023).

We would like to express our gratitude to the authors of these works for their valuable insights and contributions to the field of large language model training and data synthesis.
