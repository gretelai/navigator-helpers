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
Generate a natural language prompt that describes a Python coding task.

### Instructions:
    * Use a code complexity of "{complexity}".
    * Imagine the code will be used in the "{domain}" domain.
    * Write code that might be used in a "{topic}" context.
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
Create the contents of a Python requirements.txt file with dependencies for a "{domain}" project.

### Instructions:
    * Assume the project is related to "{topic}". 
    * Do not include package version numbers.
    * Do not include any comments.
    * Limit the number of dependencies to the most common ones.
    * Do not exceed {max_dependencies} dependencies.
    * Enclose the package names in triple backticks.

### Example:

```
package1
package2
package3
```
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
Generate a natural language prompt that describes an SQL query. Base your prompt the following 
CREATE statements for tables and/or views:

### CREATE statements:
{sql_context}

### Instructions:
    * Provide a well-formulated question or command in everyday English, representing a user query to a database.
    * The response to your prompt should require SQL of complexity "{complexity}".
    * The prompt should be in the "{domain}" domain and pertain to "{topic}".
    * Return only the natural language prompt without any code or other comments.
    
### Prompt:
""",
    sql_code_generation="""\
### Natural Language Prompt:
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
    
Write SQL statements that create tables and views that already exist in a database

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
