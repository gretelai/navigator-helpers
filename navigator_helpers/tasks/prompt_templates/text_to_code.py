NL_TYPE_PYTHON = {
    "prompt": "Generate a natural language prompt for a Python coding task.",
    "description": "Write a natural language description of a particular Python program.",
    "instruction": "Produce an instruction that instructs a user to write Python code for a specific task.",
    "question": "Ask a question about how to solve a problem using a Python program.",
}
NL_TYPE_SQL = {
    "prompt": "Generate a natural language prompt for an SQL query.",
    "description": "Write a natural language description of a particular SQL query.",
    "instruction": "Produce an instruction that instructs a user to write an SQL query.",
    "question": "Ask a question that can be answered with an SQL query.",
}


nl2python_template_dict = dict(
    domains="""\
Create a list of {num_domains} unique industries where you expect to find software engineers who code in Python. 

### Instructions:
    * Do not use abbreviations.
    * Keep each industry name to 1-5 words, preferring concise names.
    * List the industries in a valid JSON array.
""",
    topics_from_domains="""\
Create a list of {num_topics} topics that are associated with software in the following domain: {domain}

### Instructions:
    * Do not use abbreviations.
    * Keep each topic name to 1-5 words, preferring concise names
    * List the topics in a valid JSON array.
""",
    complexity="""\
Come up with a list of {num_levels} complexity levels for software in the Python programming language.

### Instructions:
    * Each complexity level should be a short description of the level of complexity.
    * Do not mention specific libraries or frameworks.
    * List the levels in a valid JSON array.
    
#### Example:
    '["Beginner: Basic syntax and data types", "Intermediate: Functions and control structures", '
    '"Advanced: Object-oriented programming and error handling"]'
""",
    python_natural_language="""\
{nl_type_description}

### Instructions:
    * The response to your {nl_type} should require code of complexity "{complexity}".
    * Your {nl_type} should be in the "{domain}" domain and pertain to "{topic}".
    * Return only the natural language prompt without any code or other comments.
    
### Prompt:
""",
    python_code_generation="""\
{python_natural_language}

### Instructions
    * The code should have a complexity of "{complexity}".
    * Write code that might be used in the "{domain}" industry within a "{topic}" context.
    * Try to include at least 1 of the following Python packages: {suggested_packages}.
    * Include ONLY a SINGLE block of code without any additional text.
""",
    python_suggested_packages="""\
Write the contents of the requirements.txt file that would be used for a Python project in the "{domain}" domain.

### Instructions:
    * Include up to {max_dependencies} packages.
    * Do not include package version numbers.
    * Limit the dependencies to packages that are commonly used in the "{domain}" domain.
    * List only the package names, without any additional text.

### Example requirements.txt:
```
package1
package2
package3
package4
package5
```

### Output requirements.txt:
""",
)


nl2sql_template_dict = dict(
    domains="""\
Create a list of {num_domains} unique industries where you expect to find professionals who write SQL queries.

### Instructions:
    * Do not use abbreviations.
    * Keep each industry name to 1-5 words, preferring concise names.
    * List the industries in a valid JSON array.
""",
    topics_from_domains="""\
Create a list of {num_topics} topics that are associated with writing SQL in the following domain: {domain}

### Instructions:
    * Do not use abbreviations.
    * Keep each topic name to 1-5 words, preferring concise names
    * List the topics in a valid JSON array.
""",
    complexity="""\
Come up with a list of {num_levels} SQL techniques/concepts of increasing complexity, from basic to advanced.

### Instructions:
    * Each complexity level should be a short description of techniques used in SQL queries.
    * List the levels in a valid JSON array.

### Example:
'["Novice: basic SQL with a simple select statement", \
'"Intermediate: two or more joins (specify inner, outer, cross)", \
'"Advanced: aggregation functions (COUNT, SUM, AVG, MIN, MAX, etc.), and HAVING clause", \
'"Expert: window functions (e.g., ROW_NUMBER, LEAD, LAG, RANK, NTILE, PERCENT_RANK, etc.)\
with partitioning and ordering"]'
""",
    sql_natural_language="""\
{nl_type_description} Base your prompt the following CREATE statements for tables and/or views:

### CREATE statements:
{sql_context}

### Instructions:
    * The response to your {nl_type} should require SQL of complexity "{complexity}".
    * Your {nl_type} should be in the "{domain}" domain and pertain to "{topic}".
    * Return only the natural language prompt without any code or other comments.
    
### Prompt:
""",
    sql_code_generation="""\
{sql_natural_language}

### CREATE statements:
{sql_context}

### Instructions
    * Provide a complete and executable SQL query used to answer/execute the natural language prompt
    * Base the SQL on the CREATE statements above.
    * The generated SQL should have a complexity of "{complexity}".
    * Make sure there is no text of any kind preceding or following SQL code.
""",
    sql_table_and_views="""\
You are a data and SQL expert in the {domain} domain.
    
Write SQL statements that create tables and views that already exist in a database.

### Instructions
    * Include complete executable SQL table CREATE statements and/or view CREATE statements with capitalized keywords.
    * Provide up to {max_statements} tables/views that are relevant to the user's natural language prompt.
    * Table names and schemas should correspond to the "{domain}" domain within a "{topic}" context.
    * Include only SQL, without any comments or additional text.
    * Do not number the tables/views.
    * Make sure there is no text of any kind preceding or following SQL code.
""",
)

nl2code_template_dicts = {
    "python": nl2python_template_dict,
    "sql": nl2sql_template_dict,
}
