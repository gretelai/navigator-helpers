import json
import logging

from datetime import datetime
from typing import List

import numpy as np
import pandas as pd


def mix_contextual_tags(num_rows: int, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Mix contextual tags based on input dataframes.

    :param num_rows: Number of rows to generate
    :param dataframes: List of dataframes to sample from
    :return: DataFrame with mixed contextual tags
    """
    # Calculate and log the total number of unique combinations
    total_combinations = np.prod([len(df.drop_duplicates()) for df in dataframes])
    print(
        f"Total number of unique combinations possible given contextual tags: {total_combinations}"
    )

    result = pd.DataFrame()
    for df in dataframes:
        if "weight" in df.columns:
            weights = df["weight"] / df["weight"].sum()
            sampled = df.sample(n=num_rows, weights=weights, replace=True)
        else:
            sampled = df.sample(n=num_rows, replace=True)

        # Remove 'weight' column if it exists
        if "weight" in sampled.columns:
            sampled = sampled.drop("weight", axis=1)

        # Append to result, preserving original column names when possible
        for col in sampled.columns:
            new_col_name = col
            counter = 1
            while new_col_name in result.columns:
                new_col_name = (
                    f"{df.name}_{col}" if hasattr(df, "name") else f"{col}_{counter}"
                )
                counter += 1
            result[new_col_name] = sampled[col].values

    # Shuffle and reset index
    result = result.sample(frac=1).reset_index(drop=True)

    # Add an 'id' column
    result.insert(0, "id", range(num_rows))

    # Log a sample of the generated tags
    print(f"Generated contextual tags (sample of 5 rows):\n{result.head()}")

    return result


def batch_and_write_data(generator, contextual_tags, batch_size, file_prefix):
    num_batches = len(contextual_tags) // batch_size

    logging.info(
        f"Beginning data generation. {num_batches} batches of size {batch_size}."
    )
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = start_index + batch_size
        batch_tags = contextual_tags.iloc[start_index:end_index]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{file_prefix}_batch_{batch_num + 1}_{timestamp}.jsonl"
        logging.info(f"Writing batch {batch_num}/{num_batches} to {filename}")

        synthetic_data = generator.generate_data(batch_tags)
        with open(filename, "w") as f:
            for item in synthetic_data:
                json.dump(item, f)
                f.write("\n")

        print(f"Batch {batch_num + 1}/{num_batches} complete. Output file: {filename}")

    print(f"Synthetic data generation complete. {num_batches} files generated.")
