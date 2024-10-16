# Script with all the classes and tests implemented
from collections import Counter
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pandas import Series
from pandas.core.dtypes.common import is_integer_dtype, is_numeric_dtype
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

    def __init__(
        self,
        llm_suite: GretelLLMSuite,
        dataset: pd.DataFrame,
        code_lang: Optional[str] = None,
        eval_kwargs: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(llm_suite)
        self.dataset = dataset
        self.llm_suite = llm_suite
        self.code_lang = code_lang
        self.eval_kwargs = eval_kwargs
        self.output_dataset = None

    def _determine_column_data_type(self, column: Series) -> str:
        """
        Similar to https://github.com/Gretellabs/monogretel/blob/110622b0c8131f0c1a1443a8e25722a92857c3d2/python/src/gretel_core/harbor/artifacts/analyzers/field_features.py#L208C1-L281C20

        Between the NavFT definition and the SQS Report definition of categorical variables,
        here we adopt the SQS definition, where the key difference is the percentage of maximum allowed unique
        values stays a constant 10% as dataset size increases, compared to the NavFT definition where that percentage
        shrinks as the dataset size gets really large, like 1 million records.
        The reasoning here is that the number of unique values, eg. data seeds,
        should proportionally increase as the dataset size increases.
        """
        # If on average each entry has more than one space, we consider it a text field
        _TEXT_FIELD_AVG_SPACE_COUNT_THRESHOLD = 0.1

        non_na_data = column.dropna()
        non_na_count = int(non_na_data.count())
        unique_count = int(non_na_data.nunique())

        if non_na_count == 0:
            # empty fieldn
            return "Other"

        diff = non_na_count - unique_count
        diff_percent = diff / non_na_count

        space_count = sum(str(entry).strip().count(" ") for entry in non_na_data)

        if is_numeric_dtype(non_na_data.dtype):
            # We can visualize numeric data with histograms, but we will not use it for diversity calculations
            min_value = int(non_na_data.min())
            if unique_count <= 10 and min_value >= 0:
                return "Categorical"
            if unique_count == non_na_count and is_integer_dtype(non_na_data.dtype):
                # All unique integer values, potentially an ID field
                return "Other"
            return "Numeric"

        if diff_percent >= 0.9 or (diff_percent >= 0.7 and len(non_na_data) <= 50):
            return "Categorical"

        if space_count / non_na_count > _TEXT_FIELD_AVG_SPACE_COUNT_THRESHOLD:
            # Count datetime fields as "Other"
            try:
                # pd.to_datetime(non_na_data)
                pd.to_datetime(non_na_data, format="%Y-%m-%d")
                return "Other"
            except:
                pass
            return "Text"

        return "Other"

    def _get_tfidf_vectors(self, column: Series) -> np.ndarray:
        vectorizer = TfidfVectorizer().fit_transform(column)
        return vectorizer.toarray()

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
        # text_columns = self.dataset.select_dtypes(include=[object, 'string'])
        concatenated_text = text_columns.apply(
            lambda row: " ".join(row.values.astype(str)), axis=1
        )
        overall_vectors = self._get_tfidf_vectors(concatenated_text)
        cosine_sim = cosine_similarity(overall_vectors)
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
        Given a hash { 'species': count }, returns the Simpson Diversity Index.
        Suitable for categorical data.
        Returns:
            - Simpson Diversity Index in the range [0, 1), where higher value indicates higher diversity
        """

        def p(n, N):
            return float(n) / N if n != 0 else 0

        N = sum(data.values())
        return 1 - sum(p(n, N) ** 2 for n in data.values() if n != 0)

    def feature_distribution(self):
        """
        Evaluation test for feature distribution, providing distributions and diversity scores
        for categorical, numerical, and text columns.

        Returns:
            - Dictionary where each column is mapped to a distribution
            - Dictionary where each column is mapped to a diversity score (if applicable)
        """
        results = {}
        distribution = {}
        score = {}
        column_data_types = {}

        # Iterate through each column to calculate appropriate distributions
        for col in self.dataset.columns:
            column_data_type = self._determine_column_data_type(self.dataset[col])
            column_data_types[col] = column_data_type

            if column_data_type == "Categorical":
                # Distribution and diversity for categorical columns
                col_value_counts = self.dataset[col].value_counts().to_dict()
                gini_simpson_index = self.simpson_diversity_index(data=col_value_counts)
                distribution[col] = col_value_counts
                score[col] = {"gini_simpson_index": round(gini_simpson_index, 4)}

            elif column_data_type == "Numeric":
                # Distribution for numerical columns - using histogram bins and basic stats
                col_histogram = np.histogram(
                    self.dataset[col].dropna(),
                    bins=10,
                    range=(self.dataset[col].min(), self.dataset[col].max()),
                )
                col_stats = {
                    "mean": self.dataset[col].mean(),
                    "median": self.dataset[col].median(),
                    "std_dev": self.dataset[col].std(),
                    "histogram": col_histogram[
                        0
                    ].tolist(),  # frequency counts in each bin
                    "bin_edges": col_histogram[1].tolist(),  # edges of bins
                }
                distribution[col] = col_stats
                score[col] = (
                    None  # Numeric columns may not have a typical 'diversity' score
                )

            elif column_data_type == "Text":
                # Text-based distributions, like length of text and average word count
                text_lengths = self.dataset[col].dropna().apply(len)
                word_counts = self.dataset[col].dropna().apply(lambda x: len(x.split()))
                avg_word_count = word_counts.mean()

                text_stats = {
                    "avg_length": text_lengths.mean(),
                    "std_length": text_lengths.std(),
                    "avg_word_count": avg_word_count,
                    "word_count_histogram": np.histogram(
                        word_counts, bins=10, range=(0, word_counts.max())
                    )[0].tolist(),
                }
                distribution[col] = text_stats
                score[col] = {
                    "text_diversity_index": self.text_diversity(self.dataset[col])
                }
            else:
                # Handling for other column types
                distribution[col] = None
                score[col] = None
        results["distribution"] = distribution
        results["score"] = score
        results["column_data_types"] = column_data_types
        return results

    def text_diversity(self, column: Series):
        """
        Given a text column, returns the text diversity index.
        It's calculated as the average cosine similarity between each record and the average embedding.
        If records are too similar with each other, the diversity index will be low.
        Returns:
            - Text diversity index in the range [0, 1), where higher value indicates higher diversity
        """

        column_vectors = self._get_tfidf_vectors(column)
        avg_column_vectors = column_vectors.mean(axis=0).reshape(1, -1)
        cosine_sim = cosine_similarity(column_vectors, avg_column_vectors)
        return 1 - cosine_sim.mean()

    def num_words_per_record(self):
        """
        Evaluation test for number of words per record in text columns.
        Returns:
            - Average number of words per record
            - Distribution of word counts per record
        """

        text_columns = self.dataset.select_dtypes(include=[object])
        # text_columns = self.dataset.select_dtypes(include=[object, 'string'])
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

        return {
            "llm_as_a_judge_scores": self.output_dataset["scores"],
            "llm_as_a_judge_mean": self.output_dataset.overall_score.mean(),
        }

    def evaluate_all(self):
        """
        Iterates through the evaluation methods and collects results with error checking.
        """
        evaluation_mapping = {
            "row_uniqueness": self.row_uniqueness,
            "feature_cardinality": self.feature_cardinality,
            "feature_distribution": self.feature_distribution,
            "num_words_per_record": self.num_words_per_record,
        }

        results = {}
        for key, func in evaluation_mapping.items():
            try:
                results[key] = func()
            except ValueError as e:
                print(f"Error in {func.__name__}: {e}")

        # Call LLM-as-a-judge evaluation on the dataset
        if self.code_lang == "sql":
            task = NL2SQLEvaluationTaskSuite(
                llm_suite=self.llm_suite, dataset=self.dataset, code_lang="sql"
            )
        elif self.code_lang == "python":
            task = NL2PythonEvaluationTaskSuite(
                llm_suite=self.llm_suite, dataset=self.dataset, code_lang="python"
            )
        else:
            task = BaseEvaluationTaskSuite(self.llm_suite, self.dataset)

        try:
            if self.eval_kwargs:
                llm_results = task.llm_as_a_judge_evaluation(**self.eval_kwargs)
            else:
                # Use default values if eval_kwargs is not provided
                default_eval_kwargs = {
                    "instruction_col_name": "instruction",
                    "code_col_name": "code",
                }
                if self.code_lang == "sql":
                    default_eval_kwargs["context_col_name"] = "sql_context"

                llm_results = task.llm_as_a_judge_evaluation(**default_eval_kwargs)
            results.update(llm_results)
        except ValueError as e:
            print(f"Error in llm_as_a_judge_evaluation: {e}")

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

        self.output_dataset["overall_score"] = self.output_dataset["scores"].apply(
            lambda x: np.mean([int(v) for k, v in x.items() if "score" in k])
        )

        return {
            "llm_as_a_judge_scores": self.output_dataset["scores"],
            "llm_as_a_judge_mean": self.output_dataset.overall_score.mean(),
        }


class VisualizationTaskSuite(BaseEvaluationTaskSuite):
    """
    Visualization class for visualizing different attributes and statistics of the dataset.
    Includes methods to generate distribution plots, heatmaps, etc.
    """

    def __init__(self, dataset: pd.DataFrame, results: dict) -> None:
        self.dataset = dataset
        self.results = results

    def plot_feature_cardinality(self):
        """
        Visualizes the cardinality of each feature in the dataset as a bar plot.
        """
        cardinality = self.results["feature_cardinality"]
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(cardinality.keys()), y=list(cardinality.values()))
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Cardinality")
        plt.title("Cardinality of Features")
        plt.show()

    def plot_row_uniqueness(self):
        """
        Visualizes row uniqueness and semantic uniqueness analysis in a single pie chart with three categories:
        Unique Rows, Partially Unique Rows, and Non-Unique Rows.
        """
        if "row_uniqueness" not in self.results:
            raise ValueError("Row Uniqueness data is not available in the results.")

        row_uniqueness = self.results["row_uniqueness"]

        # Define categories for the pie chart
        unique_rows = row_uniqueness["percent_unique"]
        semantically_unique_only = abs(
            row_uniqueness["percent_semantically_unique"] - unique_rows
        )
        non_unique_rows = 100 - row_uniqueness["percent_semantically_unique"]

        # Data for the pie chart
        labels = [
            "Unique Rows",
            "Partially Unique Rows (Semantically Unique Only)",
            "Non-Unique Rows",
        ]
        sizes = [unique_rows, semantically_unique_only, non_unique_rows]

        # Colors for the pie chart
        colors = ["#66b3ff", "#99ff99", "#ff9999"]

        wedges, texts, autotexts = plt.pie(
            sizes,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors,
            wedgeprops={"edgecolor": "black"},
            labeldistance=1.1,  # Fine-tune as needed
        )

        # Customize text sizes
        for text in autotexts:
            text.set_size(12)

        # Add legend
        plt.legend(wedges, labels, loc="best", bbox_to_anchor=(1, 0, 0.5, 1))

        plt.axis("equal")
        plt.title("Row Uniqueness and Semantic Uniqueness Analysis")
        plt.tight_layout()
        plt.show()

    def plot_feature_distribution(self):
        """
        Plots feature distributions for categorical, numerical, and text columns
        in a single window with multiple subplots using distribution data from self.results.

        Assumes self.results['distribution'] contains per-column distribution information
        and self.results['score'] contains relevant diversity or statistical scores.
        """
        # Extract distribution and score data from self.results
        distribution = self.results["feature_distribution"]["distribution"]
        score = self.results["feature_distribution"]["score"]

        num_cols = len(distribution)
        num_rows = (
            num_cols + 2
        ) // 3  # Define rows to accommodate all columns in a 3-column layout

        fig, axs = plt.subplots(num_rows, 3, figsize=(20, 5 * num_rows))
        fig.suptitle("Feature Distributions", fontsize=16)

        # Flatten axes array to handle as a 1D array
        axs = axs.flatten()

        for i, (col, dist) in enumerate(distribution.items()):
            # Determine the data type based on the available statistics
            column_data_type = self._determine_column_data_type(self.dataset[col])

            if column_data_type == "Categorical" and dist:
                # Plot categorical distributions as a bar chart without `palette`
                categories = list(dist.keys())
                frequencies = list(dist.values())
                sns.barplot(x=categories, y=frequencies, ax=axs[i])
                axs[i].set_xticks(range(len(categories)))  # Set x-ticks first
                axs[i].set_xticklabels(categories, rotation=45, ha="right", fontsize=10)
                axs[i].set_xlabel("Category", fontsize=10)
                axs[i].set_ylabel("Frequency", fontsize=10)
                axs[i].set_title(f"{col} (Categorical)", fontsize=12)

            elif column_data_type == "Numeric" and dist:
                # Plot histogram for numeric columns using stored histogram data
                hist_data = dist.get("histogram", [])
                bin_edges = dist.get("bin_edges", [])
                if hist_data and bin_edges:
                    axs[i].hist(
                        bin_edges[:-1],
                        bins=bin_edges,
                        weights=hist_data,
                        color="skyblue",
                        edgecolor="black",
                    )
                    axs[i].set_xlabel("Value", fontsize=10)
                    axs[i].set_ylabel("Frequency", fontsize=10)
                    axs[i].set_title(f"{col} (Numeric)", fontsize=12)
                    mean = dist.get("mean", "N/A")
                    median = dist.get("median", "N/A")
                    std_dev = dist.get("std_dev", "N/A")
                    axs[i].text(
                        0.7,
                        0.9,
                        f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStd Dev: {std_dev:.2f}",
                        transform=axs[i].transAxes,
                        fontsize=9,
                        verticalalignment="top",
                    )

            elif column_data_type == "Text" and dist:
                # Plot histogram for text distributions based on word count
                word_count_histogram = dist.get("word_count_histogram", [])
                axs[i].bar(
                    range(len(word_count_histogram)),
                    word_count_histogram,
                    color="salmon",
                    edgecolor="black",
                )
                axs[i].set_xlabel("Word Count Range", fontsize=10)
                axs[i].set_ylabel("Frequency", fontsize=10)
                axs[i].set_title(f"{col} (Text)", fontsize=12)
                text_diversity_score = score.get(col, {}).get(
                    "text_diversity_index", "N/A"
                )
                axs[i].text(
                    0.7,
                    0.9,
                    f"Text Diversity Index: {text_diversity_score:.4f}",
                    transform=axs[i].transAxes,
                    fontsize=9,
                    verticalalignment="top",
                )
            else:
                axs[i].set_visible(False)  # Hide empty subplots

        # Hide any remaining unused subplots
        for j in range(i + 1, len(axs)):
            axs[j].set_visible(False)

        # Adjust layout
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # def plot_num_words_per_record(self):
    #     """
    #     Visualizes the number of words per record in the dataset.
    #     """
    #     if "num_words_per_record" not in self.results:
    #         raise KeyError(
    #             "Number of words per record data is not available in the results."
    #         )

    #     num_words_data = self.results["num_words_per_record"]
    #     #average_words = num_words_data["average_words_per_record"]
    #     word_counts_per_column = num_words_data["word_counts_per_column"]
    #     average_words = float(num_words_data["average_words_per_record"])
    #     word_counts = [float(value) for value in word_counts_per_column.values()]

    #     # Plotting the word counts per column with a line for the average
    #     plt.figure(figsize=(10, 6))
    #     column_names = list(word_counts_per_column.keys())
    #     word_counts = list(word_counts_per_column.values())
    #     sns.barplot(x=column_names, y=word_counts, color="#66b3ff")

    #     # Add the average as a horizontal line
    #     plt.axhline(average_words, color="red", linestyle="--", label=f"Average words per record = ({average_words})")

    #     plt.xticks(rotation=90)
    #     plt.xlabel("Columns")
    #     plt.ylabel("Average Number of Words")
    #     plt.title("Average Number of Words per Column with Overall Average")
    #     plt.legend()

    #     plt.tight_layout()
    #     plt.show()

    def plot_num_words_per_record(self):
        """
        Visualizes the number of words per record in the dataset.
        """
        if "num_words_per_record" not in self.results:
            raise KeyError(
                "Number of words per record data is not available in the results."
            )

        num_words_data = self.results["num_words_per_record"]
        average_words = float(
            num_words_data["average_words_per_record"]
        )  # Ensure numeric type
        word_counts_per_column = num_words_data["word_counts_per_column"]

        # Convert word counts to float to ensure proper numeric handling
        column_names = list(word_counts_per_column.keys())
        word_counts = [float(value) for value in word_counts_per_column.values()]

        # Plotting the word counts per column with a line for the average
        plt.figure(figsize=(10, 6))
        sns.barplot(x=column_names, y=word_counts, color="#66b3ff")

        # Add the average as a horizontal line
        plt.axhline(
            average_words,
            color="red",
            linestyle="--",
            label=f"Average words per record = ({average_words})",
        )

        plt.xticks(rotation=90)
        plt.xlabel("Columns")
        plt.ylabel("Average Number of Words")
        plt.title("Average Number of Words per Column with Overall Average")
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_llm_as_a_judge(self):
        """
        Visualizes the LLM-as-a-judge evaluation scores for each criterion.
        """
        # TODO: Do discrete scores because LLM-as-a-judge scores are discrete
        if "llm_as_a_judge_scores" not in self.results:
            raise ValueError(
                "LLM-as-a-judge evaluation data is not available in the results."
            )

        # Set the specific bins for discrete values from 0 to 4
        bins = [0, 1, 2, 3, 4, 5]  # Include 5 to cover the upper edge of the last bin

        llm_scores = self.results["llm_as_a_judge_scores"]
        criteria = [
            "relevance_score",
            "correctness_score",
            "readability_score",
            "scalability_score",
            "standards_score",
        ]
        # criteria = set(k for score in llm_scores for k in score if 'score' in k)
        plt.figure(figsize=(16, 10))
        for i, criterion in enumerate(criteria, 1):
            plt.subplot(3, 2, i)
            # scores = [record[criterion] for record in llm_scores]
            # Convert the scores to integers or floats
            scores = [int(record[criterion]) for record in llm_scores]
            score_counts = [scores.count(x) for x in bins]

            # Bar plot for discrete scores
            sns.barplot(x=list(bins), y=score_counts, color="red")

            # Overlay KDE plot
            sns.kdeplot(
                scores, bw_adjust=0.5, color="blue", fill=True, alpha=0.3, clip=(0, 4)
            )

            # Set x-axis limits and ticks to be discrete integers from 0 to 4
            plt.xlim(0, 4)
            plt.xticks([0, 1, 2, 3, 4])

            plt.xlabel(f'{criterion.replace("_", " ").title()}')
            plt.ylabel("Frequency")
            plt.title(f'Distribution of {criterion.replace("_", " ").title()}')

        plt.tight_layout()
        plt.show()

    def visualize_all(self):
        """
        Iterates through the results dictionary and calls the corresponding visualization
        methods based on what data is available.
        """
        visualization_mapping = {
            "feature_cardinality": self.plot_feature_cardinality,
            "row_uniqueness": self.plot_row_uniqueness,
            "feature_distribution": self.plot_feature_distribution,
            "num_words_per_record": self.plot_num_words_per_record,
            "llm_as_a_judge_scores": self.plot_llm_as_a_judge,
        }

        for key, func in visualization_mapping.items():
            if key in self.results:
                try:
                    func()
                except ValueError as e:
                    print(f"Error in {func.__name__}: {e}")
