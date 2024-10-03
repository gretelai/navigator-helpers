# Script that can be used to evaluate a synthetic dataset

# Input arguments
# - Path to the dataset
from navigator_helpers.llms.llm_suite import GretelLLMSuite

from evaluation import BaseEvaluationTaskSuite, NL2SQLEvaluationTaskSuite
from datasets import load_dataset

dataset = load_dataset("gretelai/synthetic_text_to_sql", split = "train")
dataset_10 = dataset.select(range(10))

dataset_10_pd = dataset_10.to_pandas()
# print(dataset_10_pd.columns)

llm_suite = GretelLLMSuite()
results_1 = BaseEvaluationTaskSuite(llm_suite, dataset_10_pd).row_uniqueness()
results_2 = BaseEvaluationTaskSuite(llm_suite, dataset_10_pd).feature_cardinality()
results_3 = BaseEvaluationTaskSuite(llm_suite, dataset_10_pd).feature_distribution()
results_4 = BaseEvaluationTaskSuite(llm_suite, dataset_10_pd).num_words_per_record()

dataset_1_pd = dataset.select(range(1)).to_pandas()
results_5 = NL2SQLEvaluationTaskSuite(
    llm_suite=llm_suite, dataset=dataset_1_pd, code_lang="sql"
    ).llm_as_a_critic_evaluation(
    instruction_col_name="sql_prompt", code_col_name="sql", context_col_name="sql_context"
)

results_6 = BaseEvaluationTaskSuite(llm_suite, dataset_1_pd).llm_as_a_critic_evaluation(
    instruction_col_name="sql_prompt", code_col_name="sql"
)

import pdb; pdb.set_trace()

print(results_1)
print(results_2)
print(results_3)
print(results_4)
print(results_5)
print(results_6)
