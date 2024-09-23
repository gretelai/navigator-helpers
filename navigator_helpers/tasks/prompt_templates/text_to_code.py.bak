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

nl2python_fintech_template_dict = dict(
    domains="""\
Create a list of {num_domains} unique finance-related industries where you expect to find software engineers who code in {lang}.

### Instructions:
    * Focus on industries that have significant reliance on financial data, algorithms, or transactions, where custom software solutions are often developed.
    * Examples might include industries dealing with investments, banking, insurance, financial technology (FinTech), regulatory compliance, and asset management.
    * Do not use abbreviations.
    * Keep each industry name to 1-5 words, preferring concise and specific names that clearly indicate the financial nature of the industry.
    * Avoid overly broad terms like "Finance" and instead focus on sub-industries or niches within finance.
    * List the industries in a valid JSON array.
    * Ensure the names reflect sectors where Python is particularly valuable for tasks such as data analysis, risk modeling, algorithmic trading, or financial reporting.

### Examples of acceptable responses:
    '["Investment Banking", "Insurance Underwriting", "Asset Management", "Retail Banking", "Financial Auditing"]'
""",
    topics_from_domains="""\
Create a list of {num_topics} topics that are associated with the following finance-related software domain: {domain}.

### Instructions:
    * Focus on topics that are highly relevant to the financial domain specified, covering key concepts, practices, and challenges within that industry.
    * Consider topics that involve financial modeling, data analysis, risk management, regulatory compliance, investment strategies, or transaction processing.
    * Avoid overly broad or generic topics; aim for specificity that reflects current trends and practices within the financial domain.
    * Do not use abbreviations.
    * Keep each topic name to 1-5 words, preferring concise and clear names that directly relate to the domain.
    * Ensure the topics are practical and can be addressed with software solutions, particularly using Python or other programming languages commonly used in finance.
    * List the topics in a valid JSON array.
    
### Examples of acceptable responses:
    '["Credit Risk Analysis", "Algorithmic Trading Strategies", "Portfolio Optimization", "Regulatory Reporting", "Fraud Detection"]'
""",
    complexity="""\
Come up with a list of {num_levels} complexity levels for software development in the {lang} programming language, specifically tailored for the finance industry.

### Instructions:
    * Each complexity level should be a short description of the level of complexity, focusing on tasks and challenges commonly encountered in financial software development.
    * Avoid generic descriptions; instead, tailor the complexity levels to reflect the financial domain, such as working with financial data, implementing financial algorithms, or ensuring regulatory compliance.
    * Do not mention specific libraries or frameworks, but focus on the types of tasks and problems that become more complex as skill level increases.
    * List the levels in a valid JSON array.

#### Examples of acceptable responses:
    '["Beginner: Basic financial calculations", "Intermediate: Data analysis with financial datasets", "Advanced: Implementing trading algorithms", "Expert: Developing custom financial models"]'
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
Create the contents of a Python requirements.txt file with dependencies for a {project_type} project in the finance industry.

### Instructions:
    * Focus on Python packages that are commonly used in financial software development, such as those used for data analysis, financial modeling, algorithmic trading, or regulatory compliance.
    * Do not include package version numbers.
    * Do not include any comments.
    * Limit the number of dependencies to the most essential and widely used ones in finance-related projects.
    * Do not exceed {max_dependencies} dependencies.
    * Enclose the package names in triple backticks.
    * Ensure that the selected packages are suitable for handling financial data, performing statistical analysis, or developing financial applications.

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

nl2python_healthcare_template_dict = dict(
    domains="""\
Create a list of {num_domains} unique healthcare-related industries where you expect to find software engineers who code in {lang}.

### Instructions:
    * Focus on industries that have significant reliance on patient data, medical imaging, healthcare systems, or EHR (Electronic Health Record) management, where custom software solutions are often developed.
    * Examples might include industries dealing with medical devices, health informatics, telemedicine, medical diagnostics, and healthcare analytics.
    * Do not use abbreviations.
    * Keep each industry name to 1-5 words, preferring concise and specific names that clearly indicate the healthcare nature of the industry.
    * Avoid overly broad terms like "Healthcare" and instead focus on sub-industries or niches within healthcare.
    * List the industries in a valid JSON array.
    * Ensure the names reflect sectors where Python is particularly valuable for tasks such as data analysis, medical image processing, or healthcare system automation.

### Examples of acceptable responses:
    '["Telemedicine Platforms", "Medical Imaging", "Clinical Trials Management", "EHR Systems", "Health Informatics"]'
""",
    topics_from_domains="""\
Create a list of {num_topics} topics that are associated with the following healthcare-related software domain: {domain}.

### Instructions:
    * Focus on topics that are highly relevant to the healthcare domain specified, covering key concepts, practices, and challenges within that industry.
    * Consider topics that involve healthcare data analysis, medical diagnostics, patient monitoring, regulatory compliance, or healthcare system automation.
    * Avoid overly broad or generic topics; aim for specificity that reflects current trends and practices within the healthcare domain.
    * Do not use abbreviations.
    * Keep each topic name to 1-5 words, preferring concise and clear names that directly relate to the domain.
    * Ensure the topics are practical and can be addressed with software solutions, particularly using Python or other programming languages commonly used in healthcare.
    * List the topics in a valid JSON array.
    
### Examples of acceptable responses:
    '["Medical Diagnostics", "Patient Data Analysis", "EHR Interoperability", "Telemedicine Security", "Healthcare Analytics"]'
""",
    complexity="""\
Come up with a list of {num_levels} complexity levels for software development in the {lang} programming language, specifically tailored for the healthcare industry.

### Instructions:
    * Each complexity level should be a short description of the level of complexity, focusing on tasks and challenges commonly encountered in healthcare software development.
    * Avoid generic descriptions; instead, tailor the complexity levels to reflect the healthcare domain, such as working with patient data, developing medical algorithms, or ensuring healthcare compliance.
    * Do not mention specific libraries or frameworks, but focus on the types of tasks and problems that become more complex as skill level increases.
    * List the levels in a valid JSON array.

#### Examples of acceptable responses:
    '["Beginner: Basic healthcare data management", "Intermediate: Data analysis with medical datasets", "Advanced: Implementing medical diagnostics algorithms", "Expert: Developing custom healthcare systems"]'
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
Create the contents of a Python requirements.txt file with dependencies for a {project_type} project in the healthcare industry.

### Instructions:
    * Focus on Python packages that are commonly used in healthcare software development, such as those used for medical data analysis, healthcare systems, EHR management, or healthcare diagnostics.
    * Do not include package version numbers.
    * Do not include any comments.
    * Limit the number of dependencies to the most essential and widely used ones in healthcare-related projects.
    * Do not exceed {max_dependencies} dependencies.
    * Enclose the package names in triple backticks.
    * Ensure that the selected packages are suitable for handling healthcare data, performing statistical analysis, or developing healthcare applications.

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

nl2python_education_template_dict = dict(
    domains="""\
Create a list of {num_domains} unique education-related industries where you expect to find software engineers who code in {lang}.

### Instructions:
    * Focus on industries that have significant reliance on online education platforms, educational technology, or learning management systems (LMS), where custom software solutions are often developed.
    * Examples might include industries dealing with e-learning platforms, digital content creation, educational software, learning analytics, and virtual classrooms.
    * Do not use abbreviations.
    * Keep each industry name to 1-5 words, preferring concise and specific names that clearly indicate the education nature of the industry.
    * Avoid overly broad terms like "Education" and instead focus on sub-industries or niches within education.
    * List the industries in a valid JSON array.
    * Ensure the names reflect sectors where Python is particularly valuable for tasks such as data analysis, education management, or interactive learning solutions.

### Examples of acceptable responses:
    '["E-learning Platforms", "Learning Management Systems", "Digital Content Creation", "Learning Analytics", "Virtual Classrooms"]'
""",
    topics_from_domains="""\
Create a list of {num_topics} topics that are associated with the following education-related software domain: {domain}.

### Instructions:
    * Focus on topics that are highly relevant to the education domain specified, covering key concepts, practices, and challenges within that industry.
    * Consider topics that involve online education, student data management, learning analytics, educational content creation, or interactive learning solutions.
    * Avoid overly broad or generic topics; aim for specificity that reflects current trends and practices within the education domain.
    * Do not use abbreviations.
    * Keep each topic name to 1-5 words, preferring concise and clear names that directly relate to the domain.
    * Ensure the topics are practical and can be addressed with software solutions, particularly using Python or other programming languages commonly used in education.
    * List the topics in a valid JSON array.
    
### Examples of acceptable responses:
    '["Student Data Analysis", "Adaptive Learning Systems", "Online Course Creation", "Interactive Learning Tools", "Learning Analytics"]'
""",
    complexity="""\
Come up with a list of {num_levels} complexity levels for software development in the {lang} programming language, specifically tailored for the education industry.

### Instructions:
    * Each complexity level should be a short description of the level of complexity, focusing on tasks and challenges commonly encountered in education software development.
    * Avoid generic descriptions; instead, tailor the complexity levels to reflect the education domain, such as building online learning platforms, managing student data, or creating interactive learning experiences.
    * Do not mention specific libraries or frameworks, but focus on the types of tasks and problems that become more complex as skill level increases.
    * List the levels in a valid JSON array.

#### Examples of acceptable responses:
    '["Beginner: Basic student data management", "Intermediate: Data analysis with educational datasets", "Advanced: Implementing adaptive learning systems", "Expert: Developing custom education platforms"]'
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
Create the contents of a Python requirements.txt file with dependencies for a {project_type} project in the education industry.

### Instructions:
    * Focus on Python packages that are commonly used in education software development, such as those used for student data analysis, learning management, or interactive learning tools.
    * Do not include package version numbers.
    * Do not include any comments.
    * Limit the number of dependencies to the most essential and widely used ones in education-related projects.
    * Do not exceed {max_dependencies} dependencies.
    * Enclose the package names in triple backticks.
    * Ensure that the selected packages are suitable for handling educational data, creating online learning platforms, or developing interactive learning tools.

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
