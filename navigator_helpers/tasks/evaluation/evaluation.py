# Script with all the classes and tests implemented
import pandas as pd

class BaseEvaluationTaskSuite():
    # Generic class for evaluation task suites
    # Tests include:
        # 1. Row uniqueness
        # 2. Feature cardinality
        # 3. Feature distribution
        # 4. num_words per record
        # 5. LLM-as-a-critic evaluation based on a generic dataset rubric (e.g. data quality, relevance, correctness, readability)

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset
    
    def row_uniqueness(self):
        # Test for brute force row uniqueness / semanantic uniqueness
            # Input : Pandas dataset
            # Output : Percentage of rows are unique,
                        # Percentage of rows are semantically unique
                        # Ids of rows that are not unique
                        # Ids of rows that are not unique semantically
            # Assumptions : N/A
            # Algorithm for row uniqueness:
                # 1. df.unique on the whole dataset, if the length of the unique rows is equal to the length of the dataset, then all rows are unique
                # 2. Group by contextual tags and then check for uniqueness using df.unique
                # 4. Group by prompts and then check for uniqueness using df.unique
                # 5. Check fuzzy match for other columns
            # Improvements :
                # Fuzzy matching
                # Word-llama
                # Embeddings to check for cosine similarity
            # Example :
                # Prompt 1 : "How to implement k-means clustering in Python?"
                # Prompt 2 : "Can you implement k-means clustering in Python?"
            # Example 2:
                # Response 1 : "def square_root(num): ..."
                # Response 2 : "def square_root_of(num): ..."
        pass

    def feature_cardinality(self):
        # Test for feature cardinality
            # Input : Pandas dataset
            # Output : Returns a dictionary where each column is mapped to cardinality
            # Algorithm for feature cardinality:
                # 1. Calculate the percentage of unique values in each column using df.unique
                # 2. Additional logging to show the distribution of unique values
            # Related : https://github.com/Gretellabs/monogretel/blob/master/python/src/gretel_tabllm/astrolabe/expect/data.py#L117
        cardinality = {col: self.dataset[col].nunique() / len(self.dataset) for col in self.dataset.columns}
        return cardinality

    def feature_distribution(self):
        # Test for feature distribution
            # Input : Pandas dataset
            # Output : Returns a dictionary where each column is mapped to a distribution
            # Algorithm for feature distribution:
                # 1. Calculate the distribution of values in each column using df.value_counts
                # 2. Need to think about categorical, numerical and text columns
                # 3. Example: For code column, we can take the distribution of code length
            # Related : Look at HF dataset viualization
        #TODO: Some preprocessing
        distribution = {}
        for col in self.dataset.columns:
            distribution = self.dataset[col].value_counts()
        return distribution

    def _num_words_per_record(self):
        # Test for number of words per record
            # Input : Pandas dataset
            # Output : Returns a dictionary where each record is mapped to the count of words
            # Plan to use this private function for feature distribution
        pass

    def llm_as_a_critic_evaluation(self):
        # Test for LLM-as-a-critic evaluation based on a generic dataset rubric
        # Generic dataset rubric includes:
            # 1. Diversity
            # 2. Relevance
            # 3. Correctness
            # 4. Difficulty
            # Want this rubric to be complimentary to linter-based scoring
            # Related : https://github.com/gretelai/navigator-helpers/blob/main/navigator_helpers/tasks/prompt_templates/llm_as_a_judge.py
        # We already have this, need to integrate it here
        pass

# Check if validation was performed and if that column exists in the dataset
# Let the user know you are performing validation by logging
# Evaluate only the valid records
# For now, we have only code specific validation for python & SQL
# Report scores only on valid records
# Validation functions can be seperate but should be called from the evaluation task suite

class NL2CodeEvaluationTaskSuite(BaseEvaluationTaskSuite):
    # Class for evaluation task suites for natural language to code tasks
    # Tests include:
        # 1. LLM-as-a-critic evaluation based on code rubric (e.g. code quality, correctness, readability)
            # 1.1 Seperate rubrics for SQL and Python
        # 2. Code syntax validation
    def llm_as_a_critic_evaluation(self):
        # Test for LLM-as-a-critic evaluation based on code rubric
        pass

    def code_syntax_validation(self):
        # Test for code syntax validation
        pass

class NL2PythonEvaluationTaskSuite(BaseEvaluationTaskSuite):
    # Class for evaluation task suites for natural language to python tasks
    # Tests include:
        # 1. Linter-based scoring for python code
    def linter_based_scoring(self):
        # Validation test for linter-based scoring for python code
        pass

class NL2SQLEvaluationTaskSuite(BaseEvaluationTaskSuite):
    # Class for evaluation task suites for natural language to SQL tasks
    # Tests include:
        # 1. Linter-based scoring for SQL code
    def linter_based_scoring(self):
        # Test for linter-based scoring for SQL code
        pass
