# Navigator Helpers 🚀

Gretel Navigator is a compound AI-based system for generating high-quality synthetic data that can be used for training AI and LLMs. Navigator leverages techniques such as diverse text generation and an AI alignment process to iteratively enhance data quality. The system integrates multiple LLMs in a co-teaching and self-improvement process, generating diverse and high-quality synthetic texts while considering quality, adherence, toxicity, and bias.

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

## Usage
### Input Requirements
The input to this program is LLM training data in a pandas DataFrame format. You must specify one or more input fields, and an output field.
### Configuration
The data synthesis configuration is created using either the `SingleTextConfig` or `InstructionResponseConfig` class, depending on your use case. This includes setting the number of generations, population size, mutation rate, temperature, token limits, and specifying the API key, Navigator LLM, and co-teaching LLMs.

Example for single text generation:

```python
config = SingleTextConfig(
    input_fields=["context"],
    output_field="generated_text",
    num_generations=3,
    population_size=5,
    mutation_rate=0.5,
    temperature=0.8,
    max_tokens=150,
    api_key=GRETEL_API_KEY,
    navigator_llm=NAVIGATOR_LLM,
    co_teach_llms=CO_TEACH_LLMS,
    system_prompt=SYSTEM_PROMPT,
    format_prompt="Generate a natural text based on the given context.",
    mutation_prompt="Modify this text to make it more engaging and aligned with the context:",
    complexity_prompt="Rate the complexity of this text:",
    quality_prompt="Evaluate the quality of this text:",
    complexity_target=0.5,
    use_aaa=True,
)
```

Example for instruction-response generation:

```python
config = InstructionResponseConfig(
    input_fields=["context"],
    output_instruction_field="instruction",
    output_response_field="response",
    num_generations=3,
    population_size=5,
    mutation_rate=0.5,
    temperature=0.8,
    max_tokens=150,
    api_key=GRETEL_API_KEY,
    navigator_llm=NAVIGATOR_LLM,
    co_teach_llms=CO_TEACH_LLMS,
    system_prompt=SYSTEM_PROMPT,
    instruction_format_prompt="Generate a clear instruction based on the given context.",
    instruction_mutation_prompt="Modify this instruction to make it more specific:",
    instruction_complexity_prompt="Rate the complexity of this instruction:",
    instruction_quality_prompt="Evaluate the quality of this instruction:",
    instruction_complexity_target=0.5,
    response_format_prompt="Generate a detailed response to the given instruction.",
    response_mutation_prompt="Modify this response to make it more informative:",
    response_complexity_prompt="Rate the complexity of this response:",
    response_quality_prompt="Evaluate the quality of this response:",
    response_complexity_target=0.5,
    use_aaa=True,
)
```

### Generating synthetic text data

To run the data synthesis process, simply execute one of the example scripts:

```bash
python example_training_data.py
```

or

```bash
python example_chat.py
```

## Data Synthesis Process
The data synthesis process involves the following steps:

1. Initialization:
  - Create the appropriate configuration (SingleTextConfig or InstructionResponseConfig).
  - Initialize the EvolutionaryTextGenerator with the navigator LLM and co-teaching LLMs.
2. Text Generation:
  - Generate an initial population of texts.
  - Apply mutations to the population.
  - Filter the quality of the texts.
  - Rank the texts based on quality and complexity.
  - Select a diverse subset for the next generation.
  - Repeat for the specified number of generations.
3.  AI Align AI (AAA) Process (if enabled):
  - Apply Co-Teaching to the best text using multiple LLMs.
  - Apply Self-Teaching to further improve the text.
  - Select the final text.
4. Evaluation:
  - Evaluate the generated text for quality, relevance, and other metrics.
5. Output:
  - Return the generated text(s) or conversation.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the Gretel License. See the `LICENSE` file for details.

## Acknowledgements

This project was inspired by the techniques described in the following papers and blog posts:

- Shazeer et al. "WizardLM-2: Empowering Large Language Models with Diverse Knowledge and Progressive Learning." WizardLM 2, 15 Apr. 2024, https://wizardlm.github.io/WizardLM2/.
- Yuan et al. "Self-Rewarding Language Models." arXiv preprint arXiv:2401.10020 (2024).
- Wei et al. "StarCoder2-Instruct: Fully Transparent and Permissive Self-Alignment for Code Generation." Hugging Face Blog, 29 Apr. 2024, https://huggingface.co/blog/sc2-instruct.
- Li et al. "Textbooks Are All You Need II: phi-1.5 technical report." arXiv preprint arXiv:2309.05463 (2023).

We would like to express our gratitude to the authors of these works for their valuable insights and contributions to the field of large language model training and data synthesis.
