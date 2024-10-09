# Script with all the classes and tests implemented
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from navigator_helpers.llms.llm_suite import GretelLLMSuite
from navigator_helpers.tasks.base import BaseTaskSuite
from navigator_helpers.tasks.prompt_templates import load_prompt_template_suite
from navigator_helpers.tasks.text_to_code import utils

llm_as_a_judge_prompts = load_prompt_template_suite("llm_as_a_judge")


class BaseEvaluationTaskSuite(BaseTaskSuite):
    """
    Base class for evaluation task suites, which include a set of generic evaluation tasks :
        1. Row uniqueness
        2. Feature cardinality
        3. Feature distribution
        4. num_words per record
        5. LLM-as-a-critic evaluation based on a generic dataset rubric (e.g. data quality, relevance, correctness, readability)
    """

    def __init__(self, llm_suite: GretelLLMSuite, dataset: pd.DataFrame) -> None:
        super().__init__(llm_suite)
        self.dataset = dataset
        self.output_dataset = None

    def _get_llm_as_a_judge_prompt(self, natural_language: str, code: str) -> str:
        rubric = llm_as_a_judge_prompts.general_response_quality_rubric
        prompt = rubric(
            natural_language=natural_language,
            code=code,
        )
        return prompt

    def _eval_response_with_llm_as_a_judge(
        self,
        natural_language: str,
        code: str,
        **kwargs,
    ) -> dict:
        prompt = self._get_llm_as_a_judge_prompt(
            natural_language=natural_language, code=code, **kwargs
        )
        scores = {}
        response = utils.parse_json_str(self.llm_suite.judge_generate(prompt)) or {}
        for k, v in response.items():
            scores[f"{k}_score"] = v["score"]
            scores[f"{k}_reason"] = v["reasoning"]
        return scores

    def row_uniqueness(self):
        """
        Evaluation test for brute force row uniqueness & semantic uniqueness using TF-IDF and cosine similarity.
        Returns:
            - Percentage of rows that are unique
            - Percentage of rows that are semantically unique (fuzzy match)
            - IDs of rows that are not unique
            - IDs of rows that are not semantically unique
        Example for semantic duplicates:
            - Prompt 1 : "How to implement k-means clustering in Python?"
            - Prompt 2 : "Can you implement k-means clustering in Python?"
        """
        # TODO: Try out https://github.com/dleemiller/WordLlama for semantic uniqueness

        total_rows = len(self.dataset)

        # Brute Force row uniqueness by detecting duplicates
        unique_rows = self.dataset.drop_duplicates()
        non_unique_ids = self.dataset[self.dataset.duplicated()].index.tolist()
        percent_unique = len(unique_rows) / total_rows * 100

        # Semantic uniqueness using cosine similarity and TF-IDF
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
        """
        Evaluation test for feature cardinality
        Returns:
            - Dictionary where each column is mapped to cardinality
        """

        # Calculate the percentage of unique values in each column
        cardinality = {
            col: self.dataset[col].nunique() / len(self.dataset)
            for col in self.dataset.columns
        }
        return cardinality

    def simpson_diversity_index(self, data: dict):
        """
        Given a hash { 'species': count }, returns the Simpson Diversity Index
        Returns:
            - Simpson Diversity Index in the range [0, 1), where higher value indicates higher diversity
        """

        def p(n, N):
            """Relative abundance"""
            if n == 0:
                return 0
            else:
                return float(n) / N

        N = sum(data.values())
        return 1 - sum(p(n, N) ** 2 for n in data.values() if n != 0)

    def feature_distribution(self):
        """
        Evaluation test for feature distribution

        Returns:
            - Dictionary where each column is mapped to a distribution
        """
        # Algorithm for feature distribution:
        # 1. Calculate the distribution of values in each column using df.value_counts
        # 2. Need to think about categorical, numerical and text columns
        # 3. Example: For code column, we can take the distribution of code length
        # TODO: Implement something similar to HF dataset visualization
        # TODO: classify columns as categorical, numerical, text

        distribution = {}
        score = {}
        for col in self.dataset.columns:
            if self.dataset[col].dtype == "object":
                col_value_counts = self.dataset[col].value_counts().to_dict()
                gini_simpson_index = self.simpson_diversity_index(data=col_value_counts)
                distribution[col] = col_value_counts
                score[col] = round(gini_simpson_index, 4)
            elif self.dataset[col].dtype in ["int64", "float64"]:
                # TODO : Histogram Visualization
                pass
            else:  ## text
                # TODO : string length, word count, characters per word, etc. - use Text SQS functions?
                pass
        return distribution, score

    def num_words_per_record(self):
        """
        Evaluation test for number of words per record in text columns.
        Returns:
            - Average number of words per record
            - Distribution of word counts per record
        """

        text_columns = self.dataset.select_dtypes(include=[object])
        word_counts = text_columns.map(
            lambda x: len(str(x).split()) if isinstance(x, str) else 0
        )
        avg_words_per_record = word_counts.mean().mean()

        return {
            "average_words_per_record": avg_words_per_record,
            "word_counts_per_column": word_counts.mean().to_dict(),
        }

    def llm_as_a_judge_evaluation(
        self, instruction_col_name: str, code_col_name: str, **kwargs
    ):
        """
        Evaluation test for LLM-as-a-judge evaluation based on generic rubric
        Returns:
            - Average LLM-as-a-judge score for all records
        """

        self.output_dataset = self.dataset.copy()
        self.output_dataset["scores"] = self.dataset.apply(
            lambda row: self._eval_response_with_llm_as_a_judge(
                natural_language=row[instruction_col_name], code=row[code_col_name]
            ),
            axis=1,
        )

        # TODO: Explore a better way to aggregate scores than average
        # Calculate the average score for each record
        self.output_dataset["overall_score"] = self.output_dataset["scores"].apply(
            lambda x: np.mean([int(v) for k, v in x.items() if "score" in k])
        )

        return {"llm_as_a_judge_score": self.output_dataset.overall_score.mean()}

    def evaluate_all(self):
        results = {}
        results["row_uniqueness"] = self.row_uniqueness()
        results["feature_cardinality"] = self.feature_cardinality()
        results["feature_distribution"] = self.feature_distribution()
        results["num_words_per_record"] = self.num_words_per_record()
        return results


class NL2CodeEvaluationTaskSuite(BaseEvaluationTaskSuite):
    """
    Code evaluation class, which include a set of code specific evaluation tasks :
        1. LLM-as-a-judge evaluation based on code rubric (e.g. code quality, correctness, readability)
    """

    def __init__(
        self, llm_suite: GretelLLMSuite, dataset: pd.DataFrame, code_lang: str = ""
    ) -> None:
        super().__init__(llm_suite, dataset)
        self.code_lang = code_lang

    def _get_llm_as_a_judge_prompt(
        self, natural_language: str, code: str, **kwargs
    ) -> str:
        if self.code_lang == "sql":
            rubric = llm_as_a_judge_prompts.sql_quality_rubric
        elif self.code_lang == "python":
            rubric = llm_as_a_judge_prompts.python_quality_rubric
        else:
            rubric = llm_as_a_judge_prompts.general_code_quality_rubric
        prompt = rubric(natural_language=natural_language, code=code, **kwargs)
        return prompt


class NL2PythonEvaluationTaskSuite(NL2CodeEvaluationTaskSuite):
    """
    Python evaluation class, which include a set of python specific evaluation tasks
    """

    def linter_based_scoring(self):
        # Validation test for linter-based scoring for python code
        # TODO: Do we want to combine validation into the evaluation suite?
        pass


class NL2SQLEvaluationTaskSuite(NL2CodeEvaluationTaskSuite):
    """
    SQL evaluation class, which include a set of SQL specific evaluation tasks
    """

    def llm_as_a_judge_evaluation(
        self, instruction_col_name: str, code_col_name: str, context_col_name: str
    ):
        # LLM-as-a-critic evaluation based on code rubric
        self.output_dataset = self.dataset.copy()
        self.output_dataset["scores"] = self.dataset.apply(
            lambda row: self._eval_response_with_llm_as_a_judge(
                natural_language=row[instruction_col_name],
                code=row[code_col_name],
                sql_context=row[context_col_name],
            ),
            axis=1,
        )

        # Calculate an overall score. Use average for now. Should revisit
        self.output_dataset["overall_score"] = self.output_dataset["scores"].apply(
            lambda x: np.mean([int(v) for k, v in x.items() if "score" in k])
        )
        return {"llm_as_a_judge_score": self.output_dataset.overall_score.mean()}
