NL_TYPE_REASONING = {
    "prompt": "Generate a natural language prompt for a physical interaction question answering task.",
    "description": "Write a natural language description of a particular physical interaction question answering task",
    "instruction": "Produce an instruction that instructs a user to write physical interaction question answering task.",
    "question": "Ask a question about how to solve a problem using a physical interaction question answering task.",
}



nl2reasoning_template_dict = dict(
        objects="""\
Create a list of {num_objects} objects that are associated with physical tasks

### Instructions:
    * Do not use abbreviations
    * Keep each object name to 1-3 words
    * List the topics in a valid JSON array
""",
    domains="""\
Create a list of {num_domains} unique domains which are relevant to simple tasks involving interaction with the real world.

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
You are a test author attempting to generate questions about how to perform a task in the real world from the first-person perspective. In particular, you should generate a question that considers which of four unusual uses for a {object_} is most appropriate for a task in the "{domain}" domain within a "{topic}" context.  Explicitly follow the following steps to frame your response, using Step 1, Step 2, Step 3 etc..

1. Problem Recognition: Explicitly describe the domain.

2. Problem Representation: Construct a mental model of the domain and the object. Organize the available information, identify relevant and irrelevant details.

3. Information Retrieval: Search for relevant knowledge from memory. Recall known strategies or general rules.

4. Hypothesis Generation: Formulate possible questions, considering how the question could be posed based on existing knowledge and whether reasonable incorrect answers can be generated. When ready, generate at least three possible scenarios and four possible uses for the object in each.

5. Planning and Strategy Selection: Evaluate the potential questions and consider the competing answers for each. Weigh options and consider which could result in the best question and correct answer.

6. Reflection (Introspection): Reflects on your own thinking process, considering whether you chose the best question and answer or something else might be more appropriate.

7. Adjustment (Strategy Revision): Modify the other three possible uses to be {complexity} worse than the chosen answer.

8. Solution Selection: Once satisfied, write your desired question. Use this format:

'''
<<QUESTION:>> …
<<ANSWERS:>>
A: …
B: …
C: …
D: …

<<CORRECT ANSWER:>>  One of A, B, C or D
'''

Your test should illustrate reasoning through in the real world from the first-person perspective, as if you are performing a task. You should explicitly follow Step 1, Step 2 etc. to think through possible tasks.
    
### Prompt:
""",
    reasoning_generation="""\
{reasoning_natural_language}

### Instructions
Pretend you are given the above question with ''' ''' quotes, without the reasoning. Think through the choices step by step and describe why the correct answer is the most sensible.

""",
)

nl2reasoning_template_dicts = {
    "reasoning": nl2reasoning_template_dict
}
