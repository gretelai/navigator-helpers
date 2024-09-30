# Script with all the classes and tests implemented

class BaseEvaluationTaskSuite():
    # Generic class for evaluation task suites
    # Tests include:
        # 1. Row uniqueness
        # 2. Feature cardinality
        # 3. Feature distribution
        # 4. num_words per record
        # 5. LLM-as-a-critic evaluation based on a generic dataset rubric (e.g. data quality, relevance, correctness, readability)
    pass

class NL2CodeEvaluationTaskSuite():
    # Class for evaluation task suites for natural language to code tasks
    # Tests include:
        # 1. LLM-as-a-critic evaluation based on code rubric (e.g. code quality, correctness, readability)
            # 1.1 Seperate rubrics for SQL and Python
        # 2. Code syntax validation
    pass

class NL2PythonEvaluationTaskSuite():
    # Class for evaluation task suites for natural language to python tasks
    # Tests include:
        # 1. Linter-based scoring for python code
    pass

class NL2SQLEvaluationTaskSuite():
    # Class for evaluation task suites for natural language to SQL tasks
    # Tests include:
        # 1. Linter-based scoring for SQL code
    pass