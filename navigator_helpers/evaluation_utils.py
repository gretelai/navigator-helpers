# evaluation_utils.py
import time
from typing import Any, Dict, List

import pandas as pd

from .data_synthesis import log_message


def evaluate_texts(
    texts: List[str],
    column_name: str,
    additional_column: str,
    additional_value: str,
    format_prompt: str,
    navigator_tabular,
    max_retries: int = 3,
    verbose: bool = False,
) -> pd.DataFrame:
    text_df = pd.DataFrame(
        {column_name: texts, additional_column: [additional_value] * len(texts)}
    )

    attempt = 0
    while attempt < max_retries:
        try:
            text_scores = navigator_tabular.edit(
                prompt=f"""
                Add the following columns to the provided table:
                * instruction_score: A score from 0-100 indicating adherence to the user requested format: "{format_prompt}".
                * conformance_score: A score from 0-100 indicating the conformance of the generated text to the requested format, tags, and descriptions provided, with 100 being fully conforming and 0 being non-conforming.
                * quality_score: A score from 0-100 based on the grammatical correctness, coherence, and relevance of the generated text, with 100 being the highest quality and 0 being the lowest quality.
                * toxicity_score: A score from 0-100 indicating the level of toxic content in the generated text, with 0 being non-toxic and 100 being highly toxic.
                * bias_score: A score from 0-100 indicating the level of unintended biases in the generated text, with 0 being unbiased and 100 being heavily biased.
                * groundedness_score: A score from 0-100 indicating the level of factual correctness in the generated text, with 100 being fully grounded in facts and 0 being completely ungrounded.
                """,
                seed_data=text_df,
                disable_progress_bar=True,
            )
            for col in [
                "instruction_score",
                "conformance_score",
                "quality_score",
                "toxicity_score",
                "bias_score",
                "groundedness_score",
            ]:
                if col in text_scores:
                    text_scores[col] = text_scores[col].astype(float)
                else:
                    text_scores[col] = 0.0  # Default score if column is missing
            text_scores["average_score"] = (
                text_scores["instruction_score"] * 2
                + text_scores["conformance_score"]
                + text_scores["quality_score"]
                + (100 - text_scores["toxicity_score"])
                + (100 - text_scores["bias_score"])
                + text_scores["groundedness_score"]
            ) / 7
            return text_scores
        except KeyError as e:
            if verbose:
                log_message(f"KeyError during evaluation: {e}")
        except Exception as e:
            if verbose:
                log_message(f"Unexpected error during evaluation: {e}")

        attempt += 1
        if verbose:
            log_message(f"Retrying evaluation (attempt {attempt}/{max_retries})...")
        time.sleep(2)  # Wait before retrying

    raise Exception("Max retries exceeded during text evaluation")


def rank_texts(evaluated_texts: pd.DataFrame) -> Dict[str, Any]:
    best_idx = evaluated_texts["average_score"].idxmax()
    best_score = evaluated_texts.loc[best_idx, "average_score"]
    best_text = evaluated_texts.loc[best_idx, "text"]

    return {"text": best_text, "score": best_score, "index": best_idx}
