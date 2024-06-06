# navigator-self-align

# Creating high-quality LLM training data with Gretel and WizardLM-2 Techniques

This project presents an implementation for augmenting training data via synthetic data generation for large language models (LLMs). By leveraging techniques such as Evol-Instruct and Evol-Answer as well as AI Align AI (AAA) as described in the WizardLM-2 paper, the framework iteratively enhances data quality. Utilizing Gretel Navigator, the system integrates multiple LLMs in a co-teaching and self-improvement process, generating diverse and high-quality synthetic instructions, responses, and evaluation for quality, adherence, toxicity, and bias. This method can significantly improve and augment existing training data, facilitating the creation of robust and effective LLMs.

## Features

- **Evol-Instruct**: Generates diverse instructions based on given context and optionally provided instructions.
- **Evol-Answer**: Generates diverse responses based on given context and instructions.
- **AI Align AI (AAA)**: Optionally improves generated instructions and responses using a co-teaching and self-teaching approach.
- **Gretel Navigator Compound AI system and LLMs**: Used for synthetic data generation and evaluation of generated text.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/gretelai/navigator-self-align.git
   cd navigator-self-align
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

## Usage

### Command-line Arguments

- `--loglevel`: Set the logging level (default: `INFO`).
- `--use_aaa`: Enable AI Align AI (AAA) for improving instructions and responses (default: `True`).

### Example Command

```bash
python main.py --loglevel INFO --use_aaa
```

## Approach

### 1. Data Loading and Preprocessing

The dataset is loaded using the `datasets` or `pandas` libraries.

### 2. Instruction Generation (Evol-Instruct)

Diverse instructions are generated based on the given context and optionally provided instructions. This is done using a language model with a specified temperature and token limit.

### 3. Response Generation (Evol-Answer)

Diverse responses are generated based on the given context and instructions. Similar to instruction generation, this uses a language model with specified parameters.

### 4. AI Align AI (AAA)

Optionally, the generated instructions and responses are improved using a co-teaching and self-teaching approach. This approach will improve output quality, but adds processing time.
- **Co-Teaching**: Iteratively improves the text using multiple language models.
- **Self-Teaching**: Generates improvement suggestions and applies them to the text.

### 5. Evaluation and Selection

At each stage of generation, the texts are evaluated using a set of metrics (conformance, quality, toxicity, bias, groundedness) and the best instruction and response are selected based on these scores.

## Configuration

The input to this program is LLM training data, ideally with existing instructions and responses as examples in the domain that you wish to augment or improve.

The configuration for data augmentation can be customized by modifying the `DataAugmentationConfig` class. This includes setting the number of instructions and responses to generate, temperature, token limits, and adding fields.

Specifying one or more context (ground truth) fields is optional but highly encouraged. Specifying instruction and response fields to store the synthetically generated instructions and responses is required. If you have existing instructions and responses, they can be specified in the configuration to help guide the generation process.

Example:

```python
config = DataAugmentationConfig(
    num_instructions=5,
    num_responses=5,
    temperature=0.8,
    max_tokens_instruction=100,
    max_tokens_response=150
)
config.add_field("context", field_type="context")
config.add_field("instruction", field_type="instruction")
config.add_field("response", field_type="response")
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the Gretel License. See the `LICENSE` file for details.

## Acknowledgements

This project was inspired by the techniques described in the WizardLM-2 paper and leverages Gretel Navigator for synthetic data generation and evaluation.