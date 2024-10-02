# Script that can be used to evaluate a synthetic dataset

# Input arguments
# - Path to the dataset

from evaluation import BaseEvaluationTaskSuite
from datasets import load_dataset

dataset = load_dataset("gretelai/synthetic_text_to_sql", split = "train")
dataset_10 = dataset.select(range(10))

dataset_10_pd = dataset_10.to_pandas()
results = BaseEvaluationTaskSuite(dataset_10_pd).row_uniqueness()

print(results)
