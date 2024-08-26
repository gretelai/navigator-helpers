import json
import logging

import pandas as pd

from navigator_helpers import InstructionResponseConfig, TrainingDataSynthesizer


def main():
    """
    Main function to run the training data synthesis process
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # Gretel API configuration
    GRETEL_API_KEY = "prompt"

    NAVIGATOR_TABULAR = "gretelai/auto"
    NAVIGATOR_LLM = "gretelai/gpt-auto"
    CO_TEACH_LLMS = [
        "gretelai/gpt-llama3-1-8b",
        "gretelai/gpt-mistral-nemo-2407",
    ]  # List of co-teaching models

    # Dataset configuration
    df = pd.read_csv(
        "https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/llm-training-data/databricks_dolly_instruction_set.csv",
        nrows=10,
    )

    config = InstructionResponseConfig(
        input_fields=["context"],
        output_instruction_field="synthetic_instruction",
        output_response_field="synthetic_response",
        num_generations=3,
        population_size=5,
        mutation_rate=0.5,
        temperature=0.7,
        max_tokens=200,
        api_key=GRETEL_API_KEY,
        endpoint="https://api.gretel.ai",
        navigator_tabular=NAVIGATOR_TABULAR,
        navigator_llm=NAVIGATOR_LLM,
        co_teach_llms=CO_TEACH_LLMS,
        system_prompt="You are an expert in generating balanced, context-rich questions and comprehensive answers based on given contexts. Your goal is to create question-answer pairs that are informative, detailed when necessary, and understandable without prior knowledge, while not revealing the answer in the question.",
        instruction_format_prompt="Generate a specific and clear question directly related to a key point in the given context. The question should include enough background information to be understood without prior knowledge, while being answerable using only the information provided. Do not reveal the answer in the question. Ensure the question is focused and can be answered concisely if the information allows, but also accommodate for more detailed responses when appropriate.",
        instruction_mutation_prompt="Refine this question to include necessary context for understanding, without revealing the answer. Ensure it remains clear and can be comprehensively answered using only the information in the given context. Adjust the question to allow for a concise answer if possible, but also consider if a more detailed response is warranted based on the complexity of the topic.",
        instruction_quality_prompt="Evaluate the quality of this question based on its specificity, inclusion of necessary context, relevance to the original context, clarity for someone unfamiliar with the topic, and ability to be answered appropriately (either concisely or in detail) without revealing the answer:",
        instruction_complexity_target=3,
        response_format_prompt="Generate an informative answer to the given question. Use only the information provided in the original context. The response should be as concise as possible while fully addressing the question, including relevant context and explanations where necessary. For complex topics, provide a more detailed response. Ensure the answer provides enough background information to be understood by someone unfamiliar with the topic.",
        response_mutation_prompt="Refine this answer to balance conciseness with comprehensiveness. For straightforward questions, aim for brevity while ensuring accuracy. For complex topics, provide more detail and context. Add relevant information from the context as needed. Verify factual accuracy and correct any inaccuracies or missing key information. Ensure the answer can be understood without prior knowledge of the topic.",
        response_quality_prompt="Evaluate the quality of this answer based on its accuracy, appropriate level of detail (concise for simple questions, comprehensive for complex ones), relevance to the question, clarity for someone unfamiliar with the topic, inclusion of necessary background information, and whether it provides a satisfactory response using only the information from the given context:",
        response_complexity_target=4,
        use_aaa=True,
    )

    with open("default.json", "w") as f:
        f.write(json.dumps(config.to_dict()))

    # Create the training data synthesizer and perform synthesis
    synthesizer = TrainingDataSynthesizer(
        df,
        config,
        output_file="results.jsonl",
        verbose=True,
    )
    new_df = synthesizer.generate()

    # Print the first few rows of the synthetic data
    print(new_df.head().to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
