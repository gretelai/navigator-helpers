# Script with all the classes and tests implemented
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TODO: Add typing, docstrings

class BaseEvaluationTaskSuite:
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
        """
        Test for brute force row uniqueness / semantic uniqueness.
        Returns:
            - Percentage of rows that are unique
            - Percentage of rows that are semantically unique (fuzzy match)
            - IDs of rows that are not unique
            - IDs of rows that are not semantically unique
        """
        # Row uniqueness based on exact match
        total_rows = len(self.dataset)
        # TODO: Is df.unique or drop_duplicates better?
        unique_rows = self.dataset.drop_duplicates()
        non_unique_ids = self.dataset[self.dataset.duplicated()].index.tolist()
        percent_unique = len(unique_rows) / total_rows * 100

        # Semantic uniqueness (not exact matches) using cosine similarity and TF-IDF
        # TODO: Try out https://github.com/dleemiller/WordLlama
        # TODO: Check if this is including text columns
        text_columns = self.dataset.select_dtypes(include=[object])
        concatenated_text = text_columns.apply(
            lambda row: " ".join(row.values.astype(str)), axis=1
        )
        vectorizer = TfidfVectorizer().fit_transform(concatenated_text)
        vectors = vectorizer.toarray()

        cosine_sim = cosine_similarity(vectors)
        np.fill_diagonal(cosine_sim, 0)
        cosine_sim[np.tril_indices(cosine_sim.shape[0])] = 0
        non_semantically_unique_ids = np.where((cosine_sim > 0.85) & (cosine_sim < 1.0))
        non_semantically_unique_ids = list(
            zip(non_semantically_unique_ids[0], non_semantically_unique_ids[1])
        )
        percent_semantically_unique = (
            (total_rows - len(non_semantically_unique_ids)) / total_rows * 100
        )

        return {
            "percent_unique": percent_unique,
            "percent_semantically_unique": percent_semantically_unique,
            "non_unique_ids": non_unique_ids,
            "non_semantically_unique_ids": non_semantically_unique_ids,
        }

    def feature_cardinality(self):
        # Test for feature cardinality
        # Input : Pandas dataset
        # Output : Returns a dictionary where each column is mapped to cardinality
        # Algorithm for feature cardinality:
        # 1. Calculate the percentage of unique values in each column using df.unique
        # 2. TODO: Additional logging to show the distribution of unique values
        # Related : https://github.com/Gretellabs/monogretel/blob/master/python/src/gretel_tabllm/astrolabe/expect/data.py#L117
        cardinality = {
            col: self.dataset[col].nunique() / len(self.dataset)
            for col in self.dataset.columns
        }
        return cardinality

    def simpson_di(self, data: dict):
        """Given a hash { 'species': count } , returns the Simpson Diversity Index"""

        def p(n, N):
            """Relative abundance"""
            if n == 0:
                return 0
            else:
                return float(n) / N

        N = sum(data.values())

        return sum(p(n, N) ** 2 for n in data.values() if n != 0)

    def feature_distribution(self):
        # Test for feature distribution
        # Input : Pandas dataset
        # Output : Returns a dictionary where each column is mapped to a distribution
        # Algorithm for feature distribution:
        # 1. Calculate the distribution of values in each column using df.value_counts
        # 2. Need to think about categorical, numerical and text columns
        # 3. Example: For code column, we can take the distribution of code length
        # Related : Look at HF dataset visualization

        # TODO: classify columns as categorical, numerical, text
        distribution = {}
        score = {}
        for col in self.dataset.columns:
            if (
                self.dataset[col].dtype == "object"
            ):  ## and some condition for categorical
                col_value_counts = self.dataset[col].value_counts().to_dict()
                gini_simpson_index = 1 - self.simpson_di(data=col_value_counts)
                distribution[col] = col_value_counts
                score[col] = round(gini_simpson_index, 4)
            elif self.dataset[col].dtype in ["int64", "float64"]:
                # TODO : Add histogram?
                pass
            else:  ## text
                # TODO : string length, word count, characters per word, etc. - use Text SQS functions?
                pass
        return distribution, score

    def num_words_per_record(self):
        # Test for number of words per record
        # Input : Pandas dataset
        # Output : Returns a dictionary where each record is mapped to the count of words
        # Plan to use this private function for feature distribution
        """
        Test for number of words per record in text columns.
        Returns the average and distribution of word counts per record.
        """
        text_columns = self.dataset.select_dtypes(include=[object])
        word_counts = text_columns.map(
            lambda x: len(str(x).split()) if isinstance(x, str) else 0
        )
        # TODO: Do we want a distribution of word counts per record?
        avg_words_per_record = word_counts.mean().mean()

        return {
            "average_words_per_record": avg_words_per_record,
            "word_counts_per_column": word_counts.mean().to_dict(),
        }

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
