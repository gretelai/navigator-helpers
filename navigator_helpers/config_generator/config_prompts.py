TAG_EXTRACTION_SYSTEM_PROMPT = """
You are an assistant that extracts contextual tags and their attributes from dataset descriptions. 
For each tag, provide diverse attributes or values. 
Ensure the attributes are specific and useful for data generation.
Minimize overlap between tags. They will be mixed together randomly to create commands for an LLM, and all mixed together versions must remain coherent. 
Present the output in JSON format as a list of objects with 'name' and 'values' fields. 
Example:
```json
[
  {
    "name": "Math Concept",
    "values": [
      "Addition", "Subtraction", "Multiplication"
    ]
  },
  {
    "name": "Problem Difficulty",
    "values": [
      "Easy", "Medium", "Hard", "Very Hard"
    ]
  }
]
```
IMPORTANT: Always return your response in this exact JSON format, not as markdown or any other format.
"""

TAG_EXTRACTION_INITIAL_USER_PROMPT = """
Dataset Description:
{description}

Extract a set of tags and their corresponding attributes or values. These tags will be used to generate diverse synthetic data by randomly combining their values. Keep the following points in mind:

1. Aim for 2-4 distinct tags, each with 5-10 attributes or values.
2. Ensure that the tags and their values can be meaningfully combined in various ways.
3. Avoid creating tags or values that would conflict with each other when randomly combined.
4. Consider the potential interactions between different tags to maintain coherence in the generated data.

Your output will be used to create instructions for synthetic data generation, so the compatibility and diversity of the tags are crucial.
"""

TAG_EXTRACTION_ITERATION_USER_PROMPT = """
Dataset Description:
{description}

Current Tags:
```json
{tags_json}
```

Expand and refine the existing set of tags to increase complexity and diversity. Your additions should complement the current tags and maintain combinatorial coherence. Please:

1. Add 3-5 new values to each existing tag, ensuring they're compatible with other tags.
2. Do not expand beyond the maximum total of 4 different tag types.
2. If there are less than 4 tags types, create 1 new tag with 5-10 values that can be meaningfully combined with the existing tags.
3. Review the entire set to ensure that random combinations of tag values will produce coherent and diverse data points.
4. If you notice any potential conflicts or inconsistencies, suggest modifications to resolve them.

Remember, these tags will be used in random combinations to generate synthetic data, so maintaining logical consistency across all possible combinations is essential.
"""

DATA_MODEL_GENERATION_PROMPT = """
Create a DataModel for the following synthetic data generation task:

Task: {user_task}

INSTRUCTIONS:

1. Field Creation:
   - Create ONLY the fields explicitly mentioned in the user's task.
   - Do NOT create any additional fields beyond what is explicitly requested or provided in the example.

2. Field Types:
   - Use 'str', 'int', 'float', or 'bool' as field types based on the task description.
   - If the field type is not specified in the task, default to 'str'.

3. Field Descriptions:
   - Provide detailed descriptions for each field.

4. Generation Instructions:
   - Customize the instructions to align precisely with the user's task.
   - **Include an example output format in the generation instructions:**
     - If the user provided an example, include it **VERBATIM** in the "Example Output" section.
     - If no user example is provided, create a synthetic example output for each field. When synthesizing text fields, be sure to create a compelling, detailed, complex synthetic text that provides all relevant background and information.


DELIVERABLE:

Provide a YAML-formatted DataModel with:
1. Basic configuration (`api_key`, `llm_model`, etc.).
2. A `generation_instructions` section that includes the task-specific instructions and an example output format.
3. A `fields` section with ONLY the explicitly requested fields, using generic field names if necessary.

Update the sample YAML template below and return a valid YAML format and object based on the provided instructions and Task.

**Sample DataModel Template:** 
```yaml
{example_yaml}
```
"""

EXAMPLE_YAML = """
# Generator configuration
api_key: prompt
llm_model: gretelai/gpt-auto
log_level: INFO
use_reflection: true
output_filename: synthetic_data.jsonl
evol_generations: 1
num_examples: 10 
generation_strategy: record

# Data model definition
fields:
  - name: field_1
    type: str
    description: |
      Provide a detailed description of `field_1` as per the task requirements.
      Explain how to incorporate relevant contextual information to enrich this field's content.
      Emphasize that any additional information should be used to generate diverse content within this field, **not as separate fields**.

  - name: field_2
    type: int
    description: |
      Provide a detailed description of `field_2` as per the task requirements.
      Detail how to use appropriate contextual information to inform the generation of this field's content, ensuring relevance and diversity.
      Stress that any contextual information should be integrated into this field's values, **not used to create additional fields**.

generation_instructions: |
  Provide a clear and concise overview of the desired dataset. Specify the required data structure, context, and content diversity. Include any relevant constraints or guidelines for generating synthetic examples.
  Example Output: 
  (Provide a sample output format for guidance on how the data should appear, ensuring it matches the described schema and context.)

# Contextual tags
contextual_tags:
  tags:
    - name: tag_1
      values:
        - value1
        - value2
        - value3
    - name: tag_2
      values:
        - value1
        - value2
        - value3

# Note: Include only the fields and sections explicitly mentioned in the task.
"""
