# Navigator Self-Align 🚀

Navigator Self-Align is a Python library for augmenting training data via synthetic data generation for large language models (LLMs). It leverages techniques such as Evol-Instruct 🌟, Evol-Answer ✨, and AI Align AI (AAA) 🧠 as described in the WizardLM-2 paper, to iteratively enhance data quality. The system integrates multiple LLMs in a co-teaching and self-improvement process, generating diverse and high-quality synthetic instructions, responses, and evaluations for quality, adherence, toxicity, and bias.

## Features

- **Evol-Instruct 🌟**: Generates diverse instructions based on given context and optionally provided instructions.
- **Evol-Answer ✨**: Generates diverse responses based on given context and instructions.
- **AI Align AI (AAA) 🧠**: Optionally improves generated instructions and responses using a co-teaching and self-teaching approach.
- **Gretel Navigator Compound AI system and LLMs 🤖**: Used for synthetic data generation and evaluation of generated text.

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

### Input Requirements 

The input to this program is LLM training data in a pandas DataFrame format. You must specify one or more context columns, an instruction column, and a response column.

### Command-line Arguments ️

- `--loglevel`: Set the logging level (default: `INFO`).
- `--no_aaa`: Disable AI Align AI (AAA) to improve runtime (default: `False`).

### Example Command 

```bash
python main.py --loglevel INFO --no_aaa
```

## Configuration ⚙️

The data augmentation configuration is created using the `DataAugmentationConfig` class. This includes setting the number of instructions and responses to generate, temperature, token limits, and specifying the Gretel API key, primary model, and maximum number of co-teaching LLMs.

Fields are added to the configuration to specify the context, instruction, and response columns.

Example:

```python
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
```

## Data Augmentation Process 🎩

1. The `DataAugmenter` class is used to perform data augmentation. It takes the preprocessed dataset, configuration, and other options such as using examples, enabling AI Align AI (AAA), and specifying the output file.

2. The `augment()` method is called to generate synthetic examples. It constructs the context based on the specified context fields, generates diverse instructions using Evol-Instruct 🌟, optionally applies AAA 🧠 to improve the instructions, selects the best instruction, generates diverse responses using Evol-Answer ✨, optionally applies AAA 🧠 to improve the responses, and selects the best response.

3. The augmented data is saved to the specified output file in CSV format and printed as JSON for further processing or analysis.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License 📜

This project is licensed under the Gretel License. See the `LICENSE` file for details.

## Acknowledgements 👏

This project was inspired by the techniques described in the WizardLM-2 paper and leverages Gretel Navigator for synthetic data generation and evaluation.