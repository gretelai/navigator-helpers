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
        pass

    """
    def expect_column_uniqueness(self, spec: dict[str, float]) -> list[ExpectResult]:
        #spec should be in the form of {"column_name": min_unique_fraction, ...}
        expect_name = "column_uniqueness"
        if precondition_result := self.data_is_present(expect_name, spec.keys()):
            return precondition_result

        results = []
        for column, min_unique_fraction in spec.items():
            if precondition_result := self.column_is_present(expect_name, column):
                results += precondition_result
                continue
            unique_fraction = len(self.observed_data[column].unique()) / len(
                self.observed_data[column]
            )
            success = unique_fraction >= min_unique_fraction
            results.append(
                self.create_expect_result(
                    name=expect_name,
                    success=success,
                    detail=self._get_detail(
                        param=column,
                        expected=f"unique_fraction >= {min_unique_fraction}",
                        found=f"{unique_fraction = }",
                        success=success,
                    ),
                )
            )
        return results
    """

    def feature_distribution(self):
        # Test for feature distribution
        pass

    def num_words_per_record(self):
        # Test for number of words per record
        pass

    def llm_as_a_critic_evaluation(self):
        # Test for LLM-as-a-critic evaluation based on a generic dataset rubric
        pass

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
        # Test for linter-based scoring for python code
        pass

class NL2SQLEvaluationTaskSuite(BaseEvaluationTaskSuite):
    # Class for evaluation task suites for natural language to SQL tasks
    # Tests include:
        # 1. Linter-based scoring for SQL code
    def linter_based_scoring(self):
        # Test for linter-based scoring for SQL code
        pass
