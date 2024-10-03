# Script that can be used to evaluate a synthetic dataset

# Input arguments
# - Path to the dataset

from datasets import load_dataset
from evaluation import BaseEvaluationTaskSuite

## TODO: add dict of datasets to test
dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
dataset_10 = dataset.select(range(1000))

dataset_10_pd = dataset_10.to_pandas()
results_1 = BaseEvaluationTaskSuite(dataset_10_pd).row_uniqueness()
# results_2 = BaseEvaluationTaskSuite(dataset_10_pd).feature_cardinality()
# results_3 = BaseEvaluationTaskSuite(dataset_10_pd).feature_distribution()
# results_4 = BaseEvaluationTaskSuite(dataset_10_pd).num_words_per_record()

print(results_1)
import pdb

pdb.set_trace()

# review specific records
# print(dataset_10_pd.loc[results_1['non_semantically_unique_ids']])

# print(results_2)
# print(results_3)
# print(results_4)
