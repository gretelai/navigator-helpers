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

### Overview

The EvolDataGenerator uses an evolutionary algorithm to produce high-quality synthetic data:

1. **Initial Population**: Generate an initial set of data based on the provided model definition.
2. **Evaluation**: Judge the quality of each data point using specified validators.
3. **Selection**: Keep the highest quality data points.
4. **Evolution**: Apply evolutionary strategies to create variations of the selected data.
5. **Iteration**: Repeat steps 2-4 for the specified number of generations.

The number of evolutionary generations is controlled by the `evol_generations` parameter in the YAML configuration:

```yaml
evol_generations: 1  # Number of evolutionary generations
```

This iterative process allows for the gradual improvement of data quality, reducing hallucinations and increasing relevance to the specified context, but with increased computational overhead. Typically 1-2 generations.

## Contextual Tags and Data Volume

The number of synthetic records created is determined by the `num_examples` parameter in the YAML configuration. Contextual tags serve as seeds for the generation process, influencing the content and characteristics of the generated data.

```yaml
num_examples: 1000  # Number of synthetic records to generate
contextual_tags:
  tags:
    - name: domain
      values:
        - Healthcare
        - Finance
        - E-commerce
      weights: [0.4, 0.3, 0.3]
```

## Evolutionary Strategies

### Using Evolutionary Strategies

You can specify evolutionary strategies for each field in your YAML configuration:

```yaml
fields:
  - name: your_field
    type: str
    description: Field description
    evolution_strategies:
      - Enhance the content with more specific details
      - Improve clarity and coherence
    evolution_rate: 0.1
```

The `evolution_rate` determines how often these strategies are applied during the evolutionary process. For best performance, focus on iteratively improving data with small changes, not outright replacing it.

## Using Validators

Validators ensure the quality and correctness of generated data. You can specify validators for each field in your YAML configuration.

### Built-in Validators

Some built-in validators include:

- `"sql:dialect"`: Validates SQL queries (supported dialects include `"postgres"`, `"ansi"`, `"bigquery"`, `"clickhouse"`, `"databricks"`, `"db2"`, `"duckdb"`,  `"hive"`, `"mysql"`, `"oracle"`, `"redshift"`, `"snowflake"`, `"soql"`, `"sparksql"`, `"sqlite"`, `"teradata"`, `"trino"`, `"tsql"`)
- `"json"`: Validates JSON structures
- `"python"`: Validates Python code

To use a built-in validator:

```yaml
fields:
  - name: sql_query
    type: str
    description: A SQL query
    validator: sql:postgres
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

## Reflection

Reflection is a process where the AI analyzes and improves its own outputs. This self-reflection capability enhances the quality of generated data by providing additional context and reasoning.

### Enabling Reflection

To enable the Reflection feature, set `use_reflection: true` in your YAML configuration:

```yaml
use_reflection: true
```

When enabled, the system will perform reflection steps during the generation process, analyzing and improving outputs at various stages.

## Detailed Examples

For more examples, see the `examples/` directory in the repository. Here's how to run some of the advanced examples:

### Example: Math and AI Reasoning Training Data

```bash
python examples/run_generation.py examples/example_advanced_gsm8k.yml
```

This example shows how to generate high-quality math problems and their solutions, using evolutionary strategies to refine and validate the generated content.

## Additional Resources

For more information on extending and customizing Gretel Navigator, please refer to the following resources:

- [Gretel AI Documentation](https://docs.gretel.ai/)