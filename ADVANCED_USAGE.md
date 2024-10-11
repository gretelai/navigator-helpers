# Advanced Usage of Gretel Navigator

This document provides detailed information on advanced features and customization options for the Gretel Navigator.

## Table of Contents

- [The Evolutionary Approach to Data Generation](#the-evolutionary-approach-to-data-generation)
- [Contextual Tags and Data Volume](#contextual-tags-and-data-volume)
- [Evolutionary Strategies](#evolutionary-strategies)
- [Using Validators](#using-validators)
- [Reflection](#reflection)
- [Detailed Examples](#detailed-examples)
- [Additional Resources](#additional-resources)

## The Evolutionary Approach to Data Generation

The EvolDataGenerator uses an evolutionary algorithm to produce high-quality synthetic data through iterative generation, evaluation, selection, and evolution steps. The number of evolutionary generations is controlled by the `evolution_generations` parameter:

```yaml
evolution_generations: 1  # Number of evolutionary generations
```

This process allows for gradual improvement of data quality, typically with 1-2 generations.

## Contextual Tags and Data Volume

Contextual tags influence the content and characteristics of the generated data:

```yaml
num_examples: 1000  # Number of synthetic records to generate
contextual_tags:
  tags:
    - name: domain
      values:
        - value: Healthcare
          weight: 0.4
        - value: Finance
          weight: 0.3
        - value: E-commerce
          weight: 0.3
```

## Evolutionary Strategies

Specify evolutionary strategies in your YAML configuration:

```yaml
evolution:
  rate: 0.1
  strategies:
    - Enhance the content with more specific details
    - Improve clarity and coherence
    - Increase the complexity and nuance of the content where appropriate
    - Make the content more diverse by introducing alternative perspectives or methods
```

The `rate` determines how often these strategies are applied during the evolutionary process.

## Using Validators

Validators ensure the quality and correctness of generated data. You can specify validators for each field in your YAML configuration.

### Built-in Validators

The following built-in validators are available:

1. SQL Validators:
   `sql:postgres`, `sql:ansi`, `sql:bigquery`, `sql:clickhouse`, `sql:databricks`, `sql:db2`, `sql:duckdb`, `sql:hive`, `sql:mysql`, `sql:oracle`, `sql:redshift`, `sql:snowflake`, `sql:soql`, `sql:sparksql`, `sql:sqlite`, `sql:teradata`, `sql:trino`, `sql:tsql`

2. Other Expert Validators:
   - `json`: Validates JSON structures
   - `python`: Validates Python code

To use a built-in validator:

```yaml
fields:
  - name: sql_query
    type: str
    description: A SQL query
    validator: sql:postgres

  - name: python_code
    type: str
    description: A Python function
    validator: python

  - name: json_data
    type: str
    description: A JSON object
    validator: json
```

### Custom Validators

For custom validation, you can provide a descriptive name, and the system will use an LLM to perform the validation based on the description:

```yaml
fields:
  - name: fortran_code
    type: str
    description: High quality Fortran code to satisfy the user request.
    validator: Fortran
```

The LLM will interpret the validator name and description to perform appropriate validation.

## Reflection

Enable reflection to allow the AI to analyze and improve its own outputs:

```yaml
use_reflection: true
```

When enabled, the system will perform reflection steps during the generation process, analyzing and improving outputs at various stages.

## Detailed Examples

For more examples, see the `examples/` directory in the repository. Here's how to run an advanced example:

```bash
python examples/run_generation.py examples/example_advanced_gsm8k.yml
```

This example generates high-quality math problems and their solutions, using evolutionary strategies to refine and validate the generated content.

## Additional Resources

For more information on extending and customizing Gretel Navigator, please refer to the following resources:

- [Gretel AI Documentation](https://docs.gretel.ai/)
- [Gretel AI Synthetic Data Generation Guide](https://docs.gretel.ai/guides/create-synthetic-data)
- [Gretel AI API Reference](https://docs.gretel.ai/reference/client-api)