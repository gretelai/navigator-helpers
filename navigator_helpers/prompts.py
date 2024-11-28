from typing import List

# Default list of evolutionary strategies
DEFAULT_EVOLUTION_STRATEGIES: List[str] = [
    "Improve the content, enhancing its quality and clarity while maintaining its core meaning.",
    "Simplify the content to make it more accessible and understandable.",
    "Increase the complexity and nuance of the content where appropriate.",
    "Make the content more diverse by introducing alternative perspectives or methods.",
]

RECORD_GENERATION_PROMPT = """
{generation_instructions}

Context:
```
{context}
```
**Important Instructions:**
- The context above contains contextual tags that should be treated as instructions for generating the record.
- Make a best effort to incorporate these contextual tags into the generated data in a natural and logical way.
- If two tags seem to conflict, choose the most logical combination or interpretation of the tags.
- Ensure that the generated content reflects the guidance provided by these contextual tags.

Generate a new record with the following fields:
{fields_description}

**Important Instructions:**

- For each field, use the format: 
  <<FIELD: field_name>>
  Field Value
  <<END_FIELD>>
- Include ALL specified fields, do not return any extra fields.
- Ensure each field value adheres to its specified type and description.
- For multi-line text or code, include it within the field delimiters without any escaping.
- Do not include any explanations or additional text outside the specified format.
- CRITICAL: ALWAYS include <<END_FIELD>> for EVERY field, including the last one.
- CRITICAL: ALWAYS include <<RECORD_COMPLETE>> at the end of each record generation.

Examples:

1. Single-line text:
<<FIELD: name>>
John Doe
<<END_FIELD>>

2. Multi-line text:
<<FIELD: description>>
This is a multi-line
description of the item.
It can span several lines.
<<END_FIELD>>

3. Numeric value:
<<FIELD: age>>
42
<<END_FIELD>>

4. Code snippet:
<<FIELD: python_function>>
def greet(name):
    return f"Hello, {{name}}!"

print(greet("World"))
<<END_FIELD>>

5. JSON data:
<<FIELD: user_preferences>>
{{
  "theme": "dark",
  "notifications": true,
  "language": "en-US"
}}
<<END_FIELD>>

Generate the record now, ensuring EVERY field ends with <<END_FIELD>> and the record ends with <<RECORD_COMPLETE>>:
"""

FIELD_GENERATION_PROMPT = """
{generation_instructions}

Context:
{context}

Current record:
{current_record}

Generate a value for the following field:
Field name: `{field_name}`
  - Type: {field_type}
  - Description: {field_description}

**Important Instructions:**

- Use the format:
  <<FIELD: {field_name}>>
  Field Value
  <<END_FIELD>>
- Ensure the value adheres to the specified type and description.
- For multi-line text or code, include it within the field delimiters without any escaping.
- Do not include any explanations or additional text outside the specified format.

Examples:

1. For a text field:
<<FIELD: bio>>
Jane Doe is a software engineer with 5 years of experience in web development.
She specializes in React and Node.js.
<<END_FIELD>>

2. For a numeric field:
<<FIELD: salary>>
75000
<<END_FIELD>>

3. For a code field:
<<FIELD: sql_query>>
SELECT users.name, orders.order_date
FROM users
JOIN orders ON users.id = orders.user_id
WHERE orders.status = 'completed'
ORDER BY orders.order_date DESC
LIMIT 10;
<<END_FIELD>>

Generate the field value now:
"""

EVOLUTION_STRATEGY_PROMPT = """
Given the following record and context, apply the evolution strategy:
"{strategy}"

Instructions:
- The evolution strategy is meant to **evolve** and **diversify** the existing generation. 
- Do **NOT** re-generate the entire data from scratch. 
- Update the existing field values and do **NOT** add any new fields.
- Use the provided evolutionary strategy to **improve** and **update** the existing generation based on the context.
- Return the updated record using the custom delimiter format without any additional explanations.

Record:
{record}

Context:
{context}

Please provide the updated record using the custom delimiter format. """

DEFAULT_SYSTEM_PROMPT = """
You are a world-class AI system, capable of complex reasoning. Reason through the query and then provide your final response. If you detect that you made a mistake in your reasoning at any point, correct yourself.
""".strip()

REFLECTION_PROMPT = """
To respond with maximum accuracy, first think about what the user is asking for, thinking step by step. During this thinking phase, have reflections that will help you clarify ambiguities. In each reflection, list the possibilities and finally choose one. Between reflections, you can think again. At the end of the thinking, draw a conclusion. Generate only the minimum text that will help you generate a better output, don't be verbose while thinking. Finally, generate an output based on the previous thinking.

Use this format for your response:
<thinking>
Think about what the user asked for.
<reflection>
This is a reflection.
</reflection>
<reflection>
This is another reflection.
</reflection>
</thinking>
<output>
Include the output here
</output>
""".strip()

MUTATION_PROMPT = """
{generation_instructions}

Apply the following evolution strategy to the given value for the `{field_name}` field:
Strategy: {evolution_strategy}
Current value: {value}
Field type: {field_type}
Field description: {field_description}
Context: {context}
Current Record (already generated fields): {current_record}

Ensure that the evolved value remains consistent with the fields that have already been generated in the current record.
Return only the evolved value for the `{field_name}` field.
"""

LLM_VALIDATOR_PROMPT = """
Validate if the following content is valid {content_type}:

```
{content}
```

If it's valid, return 'VALID'. If not, return FAIL and describe the error."""

LLM_JUDGE_PROMPT = """
As an expert judge, your task is to **always** return a PASS/FAIL evaluating the quality and relevance of the following generated record:

{generated_record}

Based on the original data model definition:

{model_definition}

Evaluate the generated record on the following criteria:
1. Adherence to instructions in the generation instructions
2. Relevance to the specified fields and their descriptions
3. Consistency with the provided context
4. Overall quality and coherence

If the record meets all criteria, respond with "PASS".
If the record fails to meet any criteria, respond with "FAIL" followed by a brief explanation of why it failed.

Your response (PASS/FAIL):"""

CONTENT_CORRECTION_PROMPT = """
Based on the original data model definition:

{model_definition}

The following {content_type} content is invalid:
Error: {error_message}
Please correct it so that it conforms to valid {content_type} syntax.
{content}
Return only the corrected version of the provided content, with no additional text, explanations, or formatting.
"""
