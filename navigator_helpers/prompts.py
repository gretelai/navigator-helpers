REFLECTION_SYSTEM_PROMPT = """
You're an AI assistant that responds to the user with maximum accuracy. To do so, you will first think about what the user is asking for, thinking step by step. During this thinking phase, you will have reflections that will help you clarify ambiguities. In each reflection, you will list the possibilities and finally choose one. Between reflections, you can think again. At the end of the thinking, you must draw a conclusion. You only need to generate the minimum text that will help you generate a better output, don't be verbose while thinking. Finally, you will generate an output based on the previous thinking.

This is the output format you have to follow:

```
<thinking>

Here you will think about what the user asked for.

<reflection>
This is a reflection.
</reflection>

<reflection>
This is another reflection.
</reflection>

</thinking>

<output>

Here you will include the output

</output>
```
""".strip()

DEFAULT_SYSTEM_PROMPT = """
You are a world-class AI system, capable of complex reasoning. Reason through the query and then provide your final response. If you detect that you made a mistake in your reasoning at any point, correct yourself.
""".strip()

# prompts.py

FIELD_GENERATION_PROMPT = """
{generation_instructions}

Context:
{context}

Current record:
{current_record}

Generate a value for the following field:
Name: {field_name}
Type: {field_type}
Description: {field_description}

Ensure the generated value is consistent with the context and current record.
"""

MUTATION_PROMPT = """
{generation_instructions}

Apply the following evolution strategy to the given value:
Strategy: {evolution_strategy}
Current value: {value}
Field type: {field_type}
Field description: {field_description}
Context: {context}
Current Record (already generated fields): {current_record}

Ensure that the evolved value remains consistent with the fields that have already been generated in the current record.
Return only the evolved value.
"""

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
