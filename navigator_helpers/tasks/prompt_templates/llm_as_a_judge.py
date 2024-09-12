llm_as_a_judge_template_dict = dict(
    sql_quality_rubric="""\
You are a SQL data expert, bringing together expertise from across data analysis, data science and data engineering.
Take a deep breath and use the data quality rubric below to grade the quality of SQL generated based on INSTRUCTIONS and CONTEXT.
Provide output in the form of a following JSON:
{{
    "relevance": {{"score": score, "reasoning": reasoning}},
    "correctness": {{"score": score, "reasoning": reasoning}},
    "readability": {{"score": score, "reasoning": reasoning}},
    "scalability": {{"score": score, "reasoning": reasoning}},
    "standards": {{"score": score, "reasoning": reasoning}}
}}

#### INSTRUCTIONS
SQL should be generated in response to the Natural Language Prompt and database context below

Natural Language Prompt:
{natural_language}

Database Context: 
{sql_context}

Generated SQL:
```sql
{code}
```

#### SQL quality rubric
Relevance: Adherence to INSTRUCTIONS and CONTEXT
* Score = 4 Perfectly meets all specified requirements.
* Score = 3 Meets most requirements with minor deviations.
* Score = 2 Moderate deviation from the instructions.
* Score = 1 Significant deviations from the instructions.
* Score = 0 Does not adhere to the instructions.

Correctness: SQL Correctness (Does the SQL query provide the correct result?)
* Score = 4 The query executes flawlessly, returning accurate and complete results as per the requirement; logic perfectly aligns with intended operations.
* Score = 3 The query returns correct results with negligible issues not affecting the main output; logic aligns well with intended outcomes. 
* Score = 2 The query returns mostly correct results but with minor inaccuracies or omissions; logic generally reflects intended operations.
* Score = 1 The query executes but returns partially correct results, significant errors present; some logical discrepancies present.
* Score = 0 The query does not execute or returns incorrect results; logic does not align with intended operations.

Readability: Readability and Maintainability (Is the SQL code easy to understand and maintain?)
* Score = 4 The code is excellently formatted and thoroughly commented, uses meaningful aliases/variable names, ensuring high readability and ease of maintenance; organizes complex queries well.
* Score = 3 The code is well-formatted and commented, making it relatively easy to understand and maintain; uses aliases and names with some organization of complex queries.
* Score = 2 The code is somewhat readable with basic formatting and some comments, but improvements are needed; needs better use of aliases/names and organization.
* Score = 1 The code has minimal formatting and few comments, making it hard to understand; lacks meaningful names and organization.
* Score = 0 The code is unreadable, with no attempt at formatting or commenting.

Scalability: Scalability (Does the solution scale well with larger datasets or more complex queries?)
* Score = 4 The solution is highly scalable, effortlessly handling large datasets and complex queries without performance degradation; avoids inefficient patterns like Cartesian joins.
* Score = 3 The solution scales well, maintaining performance with increased data volumes and complexity; minor areas for optimization.
* Score = 2 The solution is moderately scalable, handling larger datasets with some performance issues; misses some opportunities for using scalability practices.
* Score = 1 The solution shows poor scalability, with notable performance degradation under increased load; lacks effective scalability techniques.
* Score = 0 The solution does not scale; overlooks fundamental scalability practices, resulting in significant issues.

Standards: Compliance with Standards (Does the SQL query follow SQL standards and best practices?)
* Score = 4 The query strictly adheres to SQL standards and best practices, showcasing exemplary coding standards.
* Score = 3 The query closely follows SQL standards and adheres to many best practices.
* Score = 2 The query generally follows SQL standards but has room for better alignment with best practices.
* Score = 1 The query loosely follows SQL standards, with several deviations from best practices.
* Score = 0 The query does not follow SQL standards or best practices, using deprecated or non-standard syntax.
""",
    python_quality_rubric="""\
You are a Python programming expert, bringing together expertise from across software engineering, data science, and algorithmic problem-solving.
Take a deep breath and use the code quality rubric below to grade the quality of Python code generated based on INSTRUCTIONS and CONTEXT.
Provide output in the form of the following JSON:
{{
    "relevance": {{"score": score, "reasoning": reasoning}},
    "correctness": {{"score": score, "reasoning": reasoning}},
    "readability": {{"score": score, "reasoning": reasoning}},
    "efficiency": {{"score": score, "reasoning": reasoning}},
    "pythonic": {{"score": score, "reasoning": reasoning}}
}}

#### INSTRUCTIONS
The Generated Python Code should be generated in response to the Natural Language Prompt below

Natural Language Prompt:
{natural_language}

Generated Python Code
```python
{code}
```

#### Python Code Quality Rubric
Relevance: Adherence to INSTRUCTIONS and CONTEXT
* Score = 4: Perfectly meets all specified requirements.
* Score = 3: Meets most requirements with minor deviations.
* Score = 2: Moderate deviation from the instructions.
* Score = 1: Significant deviations from the instructions.
* Score = 0: Does not adhere to the instructions.

Correctness: Code Correctness (Does the Python code provide the correct result?)
* Score = 4: The code executes flawlessly, producing accurate and complete results as per the requirement; logic perfectly aligns with intended operations.
* Score = 3: The code produces correct results with negligible issues not affecting the main output; logic aligns well with intended outcomes.
* Score = 2: The code produces mostly correct results but with minor inaccuracies or omissions; logic generally reflects intended operations.
* Score = 1: The code executes but produces partially correct results, significant errors present; some logical discrepancies present.
* Score = 0: The code does not execute or produces incorrect results; logic does not align with intended operations.

Readability: Readability and Maintainability (Is the Python code easy to understand and maintain?)
* Score = 4: The code is excellently formatted, follows PEP 8 guidelines, is thoroughly commented, uses meaningful variable names, ensuring high readability and ease of maintenance; organizes complex logic well.
* Score = 3: The code is well-formatted and commented, making it relatively easy to understand and maintain; uses descriptive names and organizes logic clearly.
* Score = 2: The code is somewhat readable with basic formatting and some comments, but improvements are needed; needs better use of descriptive names and organization.
* Score = 1: The code has minimal formatting and few comments, making it hard to understand; lacks meaningful names and organization.
* Score = 0: The code is unreadable, with no attempt at formatting or commenting.

Efficiency: Efficiency and Performance (Is the code optimized for performance?)
* Score = 4: The solution is highly efficient, using appropriate data structures and algorithms; avoids unnecessary computations and optimizes for both time and space complexity.
* Score = 3: The solution is efficient, with good use of Python's built-in functions and libraries; minor areas for optimization.
* Score = 2: The solution is moderately efficient, but misses some opportunities for optimization; uses some inefficient patterns.
* Score = 1: The solution shows poor efficiency, with notable performance issues; lacks effective optimization techniques.
* Score = 0: The solution is highly inefficient; overlooks fundamental optimization practices, resulting in significant performance issues.

Pythonic: Pythonic Code and Best Practices (Does the code follow Python conventions and best practices?)
* Score = 4: The code exemplifies Pythonic principles, making excellent use of Python-specific features and idioms; follows all relevant PEPs.
* Score = 3: The code closely follows Python conventions and adheres to many best practices; good use of Python features.
* Score = 2: The code generally follows Python conventions but has room for better alignment with Pythonic practices.
* Score = 1: The code loosely follows Python conventions, with several deviations from best practices.
* Score = 0: The code does not follow Python conventions or best practices, using non-Pythonic approaches.
""", reasoning_quality_rubric="""\
You are a reasoning expert, specialising in problems involving physical interaction with the real world.
Take a deep breath and use the rubric below to grade the quality of a pair of responses to a given question based on INSTRUCTIONS and CONTEXT.

Provide output in the form of the following JSON:
{{
    "relevance": {{"score": score, "reasoning": reasoning}},
    "correctness": {{"score": score, "reasoning": reasoning}},
    "readability": {{"score": score, "reasoning": reasoning}},
    "efficiency": {{"score": score, "reasoning": reasoning}},
    "pythonic": {{"score": score, "reasoning": reasoning}}
}}

#### INSTRUCTIONS
The Generated Python Code should be generated in response to the Natural Language Prompt below

Natural Language Prompt:
{natural_language}

Generated Python Code
```python
{q_ab}
```

#### Python Code Quality Rubric
Relevance: Adherence to INSTRUCTIONS and CONTEXT
* Score = 4: Perfectly meets all specified requirements.
* Score = 3: Meets most requirements with minor deviations.
* Score = 2: Moderate deviation from the instructions.
* Score = 1: Significant deviations from the instructions.
* Score = 0: Does not adhere to the instructions.

Correctness: Code Correctness (Does the Python code provide the correct result?)
* Score = 4: The code executes flawlessly, producing accurate and complete results as per the requirement; logic perfectly aligns with intended operations.
* Score = 3: The code produces correct results with negligible issues not affecting the main output; logic aligns well with intended outcomes.
* Score = 2: The code produces mostly correct results but with minor inaccuracies or omissions; logic generally reflects intended operations.
* Score = 1: The code executes but produces partially correct results, significant errors present; some logical discrepancies present.
* Score = 0: The code does not execute or produces incorrect results; logic does not align with intended operations.

Readability: Readability and Maintainability (Is the Python code easy to understand and maintain?)
* Score = 4: The code is excellently formatted, follows PEP 8 guidelines, is thoroughly commented, uses meaningful variable names, ensuring high readability and ease of maintenance; organizes complex logic well.
* Score = 3: The code is well-formatted and commented, making it relatively easy to understand and maintain; uses descriptive names and organizes logic clearly.
* Score = 2: The code is somewhat readable with basic formatting and some comments, but improvements are needed; needs better use of descriptive names and organization.
* Score = 1: The code has minimal formatting and few comments, making it hard to understand; lacks meaningful names and organization.
* Score = 0: The code is unreadable, with no attempt at formatting or commenting.

Efficiency: Efficiency and Performance (Is the code optimized for performance?)
* Score = 4: The solution is highly efficient, using appropriate data structures and algorithms; avoids unnecessary computations and optimizes for both time and space complexity.
* Score = 3: The solution is efficient, with good use of Python's built-in functions and libraries; minor areas for optimization.
* Score = 2: The solution is moderately efficient, but misses some opportunities for optimization; uses some inefficient patterns.
* Score = 1: The solution shows poor efficiency, with notable performance issues; lacks effective optimization techniques.
* Score = 0: The solution is highly inefficient; overlooks fundamental optimization practices, resulting in significant performance issues.

Pythonic: Pythonic Code and Best Practices (Does the code follow Python conventions and best practices?)
* Score = 4: The code exemplifies Pythonic principles, making excellent use of Python-specific features and idioms; follows all relevant PEPs.
* Score = 3: The code closely follows Python conventions and adheres to many best practices; good use of Python features.
* Score = 2: The code generally follows Python conventions but has room for better alignment with Pythonic practices.
* Score = 1: The code loosely follows Python conventions, with several deviations from best practices.
* Score = 0: The code does not follow Python conventions or best practices, using non-Pythonic approaches.
""",
)
