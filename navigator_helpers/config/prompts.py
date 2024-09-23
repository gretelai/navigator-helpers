EXAMPLE_YAML = """
api_key: prompt
llm_model: gretelai/gpt-auto
log_level: INFO
use_reflection: true
output_prefix: basic_nl2sql
evol_generations: 1
num_examples: 1000

generation_instructions: |
  You are an expert in generating balanced, context-rich questions and comprehensive answers based on given contexts. Your task is to:
  1. Create questions that are specific, clear, and directly related to key points in the given context.
  2. Ensure questions include sufficient background information for understanding without prior knowledge.
  3. Craft questions that can be answered using only the information provided in the context.
  4. Generate answers that are informative, concise when possible, and detailed when necessary.
  5. Provide comprehensive responses that fully address the question, including relevant context and explanations.
  6. Maintain consistency between the question, answer, and the original context.
  7. Avoid revealing the answer in the question.
  8. Adapt the level of detail in the answer based on the complexity of the topic.

  Example Output:
    - question: "What is the capital of France, and what historical landmarks is it known for?"
    - response: "The capital of France is Paris. It is known for landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."

fields:
  - name: question
    type: str
    description: |
      Generate a specific, clear question that relates directly to a key point in the context. It should include enough background to be understood without prior knowledge, be answerable with the information given, and not reveal the answer within the question itself. The question should be concise but allow for a detailed response if needed and encourage critical thinking or analysis based on the context.

  - name: response
    type: str
    description: |
      Generate a detailed, informative answer to the question, using only the context provided. The response should be concise while fully addressing the question, include relevant background explanations as needed, and adapt detail to the topic's complexity. For more complex topics, provide deeper explanations, while ensuring consistency between the context, question, and response.
"""

TAG_EXTRACTION_SYSTEM_PROMPT = """
You are an assistant that extracts contextual tags and their attributes from dataset descriptions. 
For each tag, provide diverse attributes or values. 
Ensure the attributes are specific and useful for data generation. 
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
"""

TAG_EXTRACTION_INITIAL_USER_PROMPT = """
Dataset Description:
{description}

Extract a set of tags and their attributes. 
Aim for 3-5 different tags, each with 5-10 attributes or values.
"""

TAG_EXTRACTION_ITERATION_USER_PROMPT = """
Dataset Description:
{description}

Current Tags:
```json
{tags_json}
```

Expand on these tags and add new ones to increase complexity. 
Add 2-3 new values for existing tags and create 1-2 new tags.
"""

DATA_MODEL_GENERATION_PROMPT = """
Create a DataModel for the following synthetic data generation task:

Task: {user_task}

Contextual tags and attributes for this task:

```yaml
{tags_yaml}
```

Guidelines for generating an effective DataModel:

1. **Primary Fields**:
   - Create 2-3 primary fields that focus on generating long-form, rich content.
   - These fields should be of type `str` and designed to produce substantial, detailed text.
   - Use the contextual tags as a guide for what kind of content these fields should generate, but focus on creating fields that will result in longer, more detailed outputs.

2. **Field Descriptions**:
   - For each field, provide a detailed description that:
     a) Explains what kind of long-form content should be generated.
     b) Incorporates relevant contextual tags to guide the content generation.
     c) Emphasizes the need for depth, detail, and richness in the generated text.
   - Ensure each field has a distinct purpose that complements rather than overlaps with other fields.

3. **Generation Instructions**:
   - Craft clear instructions for generating diverse, high-quality, long-form content for each field.
   - Explain how to use the contextual tags to inform the content generation without limiting it to short, categorical responses.
   - Emphasize the creation of detailed, contextually rich content that demonstrates complex relationships or reasoning.

4. **Leveraging Contextual Tags**:
   - Use the provided contextual tags to inform the overall theme and specific details of the generated content.
   - Explain how these tags should be incorporated into the long-form content without becoming simple categorical fields themselves.

5. **Static Fields**:
   Include these unchanged:
     api_key: prompt
     data_source: null
     evol_generations: 1
     llm_model: gretelai/gpt-auto
     log_level: INFO
     num_examples: 1000
     use_reflection: true
   Set `output_prefix` to a relevant value based on the task.

Sample DataModel structure:

{example_yaml}

Provide a new DataModel in YAML format for the given task. Ensure that:
1. The model focuses on 2-3 primary fields designed for long-form content generation.
2. Field descriptions clearly guide the creation of rich, detailed content while leveraging the contextual tags.
3. Generation instructions emphasize producing diverse, high-quality, long-form content.
4. The model avoids creating separate fields for simple categorical data present in the contextual tags.
5. Each field has a distinct purpose that results in substantial, non-overlapping content.
"""
