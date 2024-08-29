# Advanced Usage of Gretel Navigator

This document provides detailed information on advanced features and customization options for the Gretel Navigator.

## The Evolutionary Approach to Data Generation

### Overview

The EvolDataGenerator uses an evolutionary algorithm to produce high-quality synthetic data:

1. **Initial Population**: Generate an initial set of data based on the provided model definition.
2. **Evaluation**: Judge the quality of each data point using specified validators.
3. **Selection**: Keep the highest quality data points.
4. **Evolution**: Apply evolutionary strategies to create variations of the selected data.
5. **Iteration**: Repeat steps 2-4 for the specified number of generations.

This iterative process allows for the gradual improvement of data quality, reducing hallucinations and increasing relevance to the specified context.

## Evolutionary Strategies

The EvolDataGenerator supports several built-in evolutionary strategies that work well for many scenarios:

- **"improve"**: Enhances the quality of the content
- **"simplify"**: Reduces complexity while maintaining core information
- **"complexity"**: Increases the sophistication of the content
- **"diversity"**: Introduces variations to create more diverse outputs

### Using Evolutionary Strategies

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

2. Reference these strategies in your `DataFieldDefinition`:

```python
DataFieldDefinition(
    name="your_field",
    type="str",
    description="Field description",
    evolutionary_strategies=["enhance", "transform"],
)
```

3. Provide the custom strategies when generating data:

```python
generator = EvolDataGenerator(config, model_def)
evolved_data = generator.generate_data(
    contextual_tags=your_contextual_data,
    output_file="output.jsonl",
    custom_evolutionary_strategies=custom_strategies
)
```

This approach allows you to define and use custom evolutionary strategies to achieve specific outcomes with your synthetic data.

## Using Validators

Validators ensure the quality and correctness of generated data. Built-in validators include:

- `"sql:dialect"`: Validates SQL queries (e.g., `"sql:postgres"`)
- `"json"`: Validates JSON structures
- `"python"`: Validates Python code

### Using Built-in Validators

To use a validator:

```python
DataFieldDefinition(
    name="sql_query",
    type="str",
    description="A SQL query",
    validator="sql:postgres",
)
```

### Custom Validation

For custom validation, provide a descriptive name, and the system will use an LLM to perform the validation based on the description:

```python
DataFieldDefinition(
    name="fortran_code",
    type="str",
    description="High quality Fortran code to satisfy the user request.",
    validator="Fortran",
)
```

## Detailed Examples

For more examples, see the `examples/` directory. Below are some advanced use cases:

### Example: Natural Language to SQL (NL2SQL)

```bash
python examples/example_nl2sql.py
```

This example demonstrates how to use the EvolDataGenerator to generate SQL queries from natural language inputs, using evolutionary strategies to improve the quality of the generated SQL.

### Example: Closed Question Answering (Closed QA)

```bash
python examples/example_closed_qa.py
```

This example shows how to generate high-quality answers to closed questions using evolutionary strategies to refine and validate the generated content.

## Additional Resources

For more information on extending and customizing Gretel Navigator, please refer to the following resources:

- [Gretel AI Documentation](https://docs.gretel.ai/)

