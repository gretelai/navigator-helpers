# Advanced Usage of Gretel Navigator

This document provides detailed information on advanced features and customization options for the Gretel Navigator.

## Table of Contents

- [The Evolutionary Approach to Data Generation](#the-evolutionary-approach-to-data-generation)
- [Contextual Tags and Data Volume](#contextual-tags-and-data-volume)
- [Evolutionary Strategies](#evolutionary-strategies)
- [Using Validators](#using-validators)
- [Reflection](#reflection)
- [Storing Reflection Outputs](#storing-reflection-outputs)
- [Detailed Examples](#detailed-examples)
- [Additional Resources](#additional-resources)

## The Evolutionary Approach to Data Generation

### Overview

The EvolDataGenerator uses an evolutionary algorithm to produce high-quality synthetic data:

1. **Initial Population**: Generate an initial set of data based on the provided model definition.
2. **Evaluation**: Judge the quality of each data point using specified validators.
3. **Selection**: Keep the highest quality data points.
4. **Evolution**: Apply evolutionary strategies to create variations of the selected data.
5. **Iteration**: Repeat steps 2-4 for the specified number of generations.

The number of evolutionary generations is controlled by the `num_generations` parameter in the `GeneratorConfig`:

```python
config = GeneratorConfig(
    api_key="your_api_key",
    llm_model="gretelai/gpt-auto",
    num_generations=3,  # Number of evolutionary generations
    evolution_rate=0.5,
    log_level="INFO",
)
```

This iterative process allows for the gradual improvement of data quality, reducing hallucinations and increasing relevance to the specified context.

## Contextual Tags and Data Volume

The number of synthetic records created is determined by the total number of contextual tags provided to the model. These tags serve as seeds for the generation process, influencing the content and characteristics of the generated data.

For example, if you provide 1000 contextual tags, the system will generate 1000 synthetic records, each influenced by its corresponding tag.

## Evolutionary Strategies

### Built-in Strategies

The EvolDataGenerator supports four built-in evolutionary strategies:

- **"simplify"**: Reduces complexity while maintaining core information
- **"diversity"**: Introduces variations to create more diverse outputs
- **"complexity"**: Increases the sophistication of the content
- **"improve"**: Enhances the quality of the content

### Using Built-in Strategies

You can use these strategies in your `DataFieldDefinition`:

```python
DataFieldDefinition(
    name="your_field",
    type="str",
    description="Field description",
    evolutionary_strategies=["improve", "complexity"],
)
```

### Custom Evolutionary Strategies

To implement custom evolutionary strategies:

1. Create a dictionary of strategy categories and their corresponding prompts:

```python
custom_strategies = {
    "enhance": [
        "Add more specific details to the content",
        "Incorporate industry-specific terminology"
    ],
    "transform": [
        "Change the tone to be more formal",
        "Rewrite the content from a different perspective"
    ]
}
```

2. Provide the custom strategies when initializing the EvolDataGenerator:

```python
generator = EvolDataGenerator(
    config,
    model_def,
    custom_evolutionary_strategies=custom_strategies
)
```

This approach allows you to define and use custom evolutionary strategies to achieve specific outcomes with your synthetic data.

## Using Validators

Validators ensure the quality and correctness of generated data. You can use built-in validators or have the LLM compose a custom validator for you.

### Built-in Validators

Some built-in validators include:

- `"sql:dialect"`: Validates SQL queries (e.g., `"sql:postgres"`)
- `"json"`: Validates JSON structures
- `"python"`: Validates Python code

To use a built-in validator:

```python
DataFieldDefinition(
    name="sql_query",
    type="str",
    description="A SQL query",
    validator="sql:postgres",
)
```

### Custom LLM-composed Validators

For custom validation, provide a descriptive name, and the system will use an LLM to perform the validation based on the description:

```python
DataFieldDefinition(
    name="fortran_code",
    type="str",
    description="High quality Fortran code to satisfy the user request.",
    validator="Fortran",
)
```

The LLM will use the provided description to compose and apply a suitable validator.

## Reflection

Reflection is a process where the AI analyzes and improves its own outputs. In Navigator, reflection is used for any generative process and works in concert with the evolutionary approach. This self-reflection capability enhances the quality of generated data by providing additional context and reasoning. It can be particularly useful for:

1. Reducing hallucinations in generated content
2. Breaking complex problems down into smaller pieces
3. Improving the overall quality and coherence of outputs
4. Creating training data with instruction tags that can be used to teach another LLM to reason

At a high level, reflection is excellent for reducing hallucinations and tackling complex problems by breaking them down into manageable components. This process allows the AI to critically evaluate its own outputs and make improvements before finalizing the generation.

### Enabling Reflection

To enable the Reflection feature, set `use_reflection=True` in your `GeneratorConfig`:

```python
config = GeneratorConfig(
    api_key="your_api_key",
    llm_model="gretelai/gpt-auto",
    num_generations=3,
    evolution_rate=0.5,
    log_level="INFO",
    use_reflection=True,
)
```

When enabled, the system will perform reflection steps during the generation process, analyzing and improving outputs at various stages.

## Storing Reflection Outputs

Storing reflection outputs, including the reflective tags, is a promising approach to teach AI more advanced reasoning capabilities. This feature allows you to capture the system's reasoning process, including intermediate steps and self-corrections.

### Enabling Storing Reflection Outputs

To enable storing reflection outputs for a specific field, set `store_full_reflection=True` in your `DataFieldDefinition`:

```python
DataFieldDefinition(
    name="your_field",
    type="str",
    description="Field description",
    evolutionary_strategies=["improve", "complexity"],
    store_full_reflection=True,
)
```

When enabled, the full reflection output for this field will be stored alongside the generated data, including the reflection tags such as `<thinking>` and `<output>`. This detailed record of the AI's reasoning process can be valuable for:

1. Analyzing the AI's decision-making process
2. Identifying areas for improvement in the AI's reasoning
3. Training other AI models to develop more advanced reasoning capabilities
4. Enhancing transparency and explainability of AI-generated content

By leveraging stored reflection outputs, researchers and developers can gain deeper insights into AI reasoning patterns and potentially use this data to create more sophisticated and reliable AI systems.

## Detailed Examples

For more examples, see the `examples/` directory. Below are some advanced use cases:

### Example: Natural Language to SQL (NL2SQL)

```bash
python examples/example_advanced_nl2sql.py
```

This example demonstrates how to use the EvolDataGenerator to generate SQL queries from natural language inputs, using evolutionary strategies to improve the quality of the generated SQL.

### Example: Grade School Math Problem Generation (GSM8k-like Logic and Reasoning)

```bash
python examples/example_advanced_gsm8k.py
```

This example shows how to generate high-quality answers to closed questions using evolutionary strategies to refine and validate the generated content.

## Additional Resources

For more information on extending and customizing Gretel Navigator, please refer to the following resources:

- [Gretel AI Documentation](https://docs.gretel.ai/)