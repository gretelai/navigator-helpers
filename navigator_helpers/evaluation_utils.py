import json
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel

from .data_synthesis import log_message
from .json_utils import parse_json_response, validate_json_keys


class Metric(BaseModel):
    name: str
    description: str
    weight: float = 1.0
    inverse: bool = False


class Metrics(BaseModel):
    metrics: List[Metric] = [
        Metric(
            name="quality",
            description="Overall quality, including grammar, coherence, and clarity",
            weight=1.0,
        ),
        Metric(
            name="relevance",
            description="How relevant the text is to the given prompt or context",
            weight=1.0,
        ),
        Metric(
            name="factual_accuracy",
            description="How factually correct and well-grounded the information is",
            weight=2.0,
        ),
        Metric(
            name="prompt_adherence",
            description="How well the text follows the provided prompt instructions",
            weight=1.0,
        ),
        Metric(
            name="bias",
            description="Level of unintended bias (0 being unbiased, 100 being heavily biased)",
            weight=1.0,
            inverse=True,
        ),
        Metric(
            name="toxicity",
            description="Presence of toxic or inappropriate content (0 being non-toxic, 100 being highly toxic)",
            weight=1.0,
            inverse=True,
        ),
    ]


def evaluate_texts(
    texts: List[str],
    llm: Any,
    prompt: Optional[str] = None,
    context: Optional[str] = None,
    max_retries: int = 3,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Evaluate a list of texts using an LLM as a judge, optionally considering provided prompt and context.

    Parameters:
    - texts (List[str]): List of texts to be evaluated.
    - llm (Any): The LLM model or interface used for judging.
    - prompt (Optional[str]): The prompt used to generate the texts (if applicable).
    - context (Optional[str]): Additional context or ground truth for evaluation.
    - max_retries (int): Maximum number of retry attempts in case of failure.
    - verbose (bool): If True, log detailed information during the process.

    Returns:
    - pd.DataFrame: DataFrame containing the original texts and evaluation scores.
    """
    metrics = Metrics()
    all_scores = []

    for idx, text in enumerate(texts):
        attempt = 0
        while attempt < max_retries:
            try:
                metrics_list = ", ".join([f"{m.name}" for m in metrics.metrics])

                evaluation_prompt = f"""Evaluate the following text based on these criteria. 
Assign a score from 0 to 100 for each criterion, using the full range of scores.
Avoid giving perfect or near-perfect scores unless the text is truly exceptional.
Consider 50 as an average score, and distribute scores above and below this benchmark based on the text's merits and flaws.

Text to evaluate: "{text}"

"""
                if prompt:
                    evaluation_prompt += f'Original prompt: "{prompt}"\n'
                if context:
                    evaluation_prompt += f'Additional context: "{context}"\n'

                evaluation_prompt += f"""
Evaluation criteria: {metrics_list}

Provide your evaluation as a JSON object with this structure:
{{
    "quality": score,
    "relevance": score,
    "factual_accuracy": score,
    "prompt_adherence": score,
    "bias": score,
    "toxicity": score
}}

Replace 'score' with your numeric score (0-100) for each criterion.
Do not include any explanations or additional text, just the JSON object.

Ensure that your scores reflect meaningful differences between strengths and weaknesses in the text.
"""
                response = llm.generate(evaluation_prompt, temperature=0.2)

                scores = parse_json_response(response, verbose)

                # Ensure all required metrics are present
                for metric in metrics.metrics:
                    if metric.name not in scores:
                        scores[metric.name] = 50  # Default to average if missing

                scores["text"] = text
                all_scores.append(scores)
                break  # Success, move to next text

            except Exception as e:
                if verbose:
                    log_message(f"Error during evaluation scoring of texts {idx + 1}: {str(e)}")

            attempt += 1
            if verbose:
                log_message(
                    f"Retrying evaluation of text {idx + 1} (attempt {attempt}/{max_retries})..."
                )
            time.sleep(2)  # Wait before retrying

        if attempt == max_retries:
            raise Exception(f"Max retries exceeded during evaluation of text {idx + 1}")

    # Combine all individual scores into a single DataFrame
    combined_scores = pd.DataFrame(all_scores)

    # Calculate composite score
    composite_score = sum(
        (
            combined_scores[m.name] * m.weight
            if not m.inverse
            else (100 - combined_scores[m.name]) * m.weight
        )
        for m in metrics.metrics
    ) / sum(m.weight for m in metrics.metrics)

    combined_scores["composite_score"] = composite_score

    return combined_scores


def rank_texts(evaluated_texts: pd.DataFrame) -> Dict[str, Any]:
    best_idx = evaluated_texts["composite_score"].idxmax()
    best_score = evaluated_texts.loc[best_idx, "composite_score"]
    best_text = evaluated_texts.loc[best_idx, "text"]

    return {"text": best_text, "score": best_score, "index": best_idx}


def relative_ranking(
    texts: List[str],
    llm: Any,
    prompt: Optional[str] = None,
    context: Optional[str] = None,
    max_retries: int = 3,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compare and rank a list of texts using an LLM as a judge, optionally considering provided prompt and context.
    If only one text is provided, it automatically receives a perfect score.

    Parameters:
    - texts (List[str]): List of texts to be compared and ranked.
    - llm (Any): The LLM model or interface used for judging.
    - prompt (Optional[str]): The prompt used to generate the texts (if applicable).
    - context (Optional[str]): Additional context or ground truth for evaluation.
    - max_retries (int): Maximum number of retry attempts in case of failure.
    - verbose (bool): If True, log detailed information during the process.

    Returns:
    - pd.DataFrame: DataFrame containing the original texts, rankings, and derived scores.
    """
    metrics = Metrics()

    # Handle the case of a single text
    if len(texts) == 1:
        return pd.DataFrame(
            {
                "text": texts,
                "rank": [1],
                "composite_score": [100.0],
                **{metric.name: [100.0] for metric in metrics.metrics},
            }
        )

    ranking_prompt = f"""Compare and rank the following texts based on these criteria:

Evaluation criteria:
{', '.join([f"{m.name}: {m.description}" for m in metrics.metrics])}

"""
    if prompt:
        ranking_prompt += f'Original prompt: "{prompt}"\n'
    if context:
        ranking_prompt += f'Additional context: "{context}"\n'

    ranking_prompt += f"""
Texts to evaluate:
{json.dumps([f"Text {i+1}: " + text[:100] + "..." for i, text in enumerate(texts)])}

Provide your ranking as a simple JSON object where the keys are the text numbers and the values are their ranks:
{{
    "1": X,
    "2": Y,
    ...
}}

Ensure that your rankings reflect meaningful differences between the texts based on the given criteria.
The best text should have rank 1, the second-best rank 2, and so on.
Provide only the JSON object, with no additional explanation.
"""

    def parse_response(response: str) -> dict:
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict) and all(
                key.isdigit() and isinstance(value, int)
                for key, value in parsed.items()
            ):
                return {int(k): v for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass

        # If JSON parsing fails, try to extract information using regex
        pattern = r'"?(\d+)"?\s*:\s*(\d+)'
        matches = re.findall(pattern, response)
        if matches:
            return {int(text): int(rank) for text, rank in matches}

        raise ValueError("Unable to parse the relative ranking response into the required format")

    attempt = 0
    while attempt < max_retries:
        try:
            response = llm.generate(ranking_prompt, temperature=0.2)

            parsed_response = parse_response(response)

            # Create DataFrame from the parsed response
            result_df = pd.DataFrame(
                [(k, v) for k, v in parsed_response.items()],
                columns=["text_number", "rank"],
            )
            result_df = result_df.sort_values("rank")

            # Add original texts
            result_df["text"] = [texts[i - 1] for i in result_df["text_number"]]

            # Calculate scores based on rankings
            num_texts = len(texts)
            result_df["composite_score"] = (
                (num_texts - result_df["rank"] + 1) / num_texts * 100
            )

            # Generate individual metric scores based on ranking
            for metric in metrics.metrics:
                if metric.inverse:
                    result_df[metric.name] = result_df["rank"] / num_texts * 100
                else:
                    result_df[metric.name] = (
                        (num_texts - result_df["rank"] + 1) / num_texts * 100
                    )

            # Reorder columns
            column_order = ["text", "rank", "text_number", "composite_score"] + [
                m.name for m in metrics.metrics
            ]
            result_df = result_df[column_order]

            return result_df

        except Exception as e:
            if verbose:
                print(f"Error during relative ranking comparison (attempt {attempt + 1}): {str(e)}")
                print(f"LLM Response:\n{response}")

        attempt += 1
        if verbose:
            print(f"Retrying relative ranking comparison (attempt {attempt}/{max_retries})...")
        time.sleep(2)  # Wait before retrying

    raise Exception("Max retries exceeded during text comparison")
