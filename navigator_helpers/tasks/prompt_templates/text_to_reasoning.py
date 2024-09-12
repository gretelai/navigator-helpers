NL_TYPE_REASONING = {
    "prompt": "Generate a natural language prompt for a physical interaction question answering task.",
    "description": "Write a natural language description of a particular physical interaction question answering task",
    "instruction": "Produce an instruction that instructs a user to write physical interaction question answering task.",
    "question": "Ask a question about how to solve a problem using a physical interaction question answering task.",
}



nl2reasoning_template_dict = dict(
    domains="""\
Create a list of {num_domains} unique domains which are relevant to tasks involving interaction with the real world.

### Instructions:
    * Do not use abbreviations.
    * Keep each domain name to 1-5 words, preferring concise names.
    * List the domains in a valid JSON array.
""",
    topics_from_domains="""\
Create a list of {num_topics} topics that are associated with physical tasks in the following domain: {domain}

### Instructions:
    * Do not use abbreviations.
    * Keep each topic name to 1-5 words, preferring concise names
    * List the topics in a valid JSON array.
""",
    complexity="""\
Come up with a list of {num_levels} complexity levels for tasks requiring physical interaction with the real world. 

### Instructions:
    * Each complexity level should be a short description of the level of complexity.
    * List the levels in a valid JSON array.
    
#### Example:
    '["Beginner: Basic logical operations and pattern matching", '
    '"Intermediate: Conditional reasoning and basic inference", '
    '"Advanced: Complex logical deductions and multi-step reasoning", '
    '"Expert: Handling ambiguity and reasoning under uncertainty"]'

""",
    reasoning_natural_language="""\
{nl_type_description}

### Instructions:
    * The response to your {nl_type} should require code of complexity "{complexity}".
    * Your {nl_type} should be in the "{domain}" domain and pertain to "{topic}".
    * Return only the natural language prompt without any code or other comments.
    
### Prompt:
""",
    reasoning_generation="""\
{reasoning_natural_language}

### Instructions
    * The reasoning task should have a complexity of "{complexity}".
    * Write reasoning that might be used in the "{domain}" domain within a "{topic}" context.
""",
)

nl2reasoning_template_dicts = {
    "reasoning": nl2reasoning_template_dict
}
