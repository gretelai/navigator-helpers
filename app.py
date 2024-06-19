import json
import logging
import time
from io import StringIO

import pandas as pd
import requests
import streamlit as st
from datasets import load_dataset
from gretel_client import Gretel

from navigator_helpers import DataAugmentationConfig, DataAugmenter, StreamlitLogHandler

# Create a StringIO buffer to capture the logging output
log_buffer = StringIO()

# Create a handler to redirect logging output to the buffer
handler = logging.StreamHandler(log_buffer)
handler.setLevel(logging.INFO)

# Set up the logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(handler)


SAMPLE_DATASET_URL = "https://gretel-public-website.s3.us-west-2.amazonaws.com/datasets/llm-training-data/dolly-examples-qa-with-context.csv"
WELCOME_MARKDOWN = """
Gretel Navigator is an interface designed to help you create high-quality, diverse training data examples through synthetic data generation techniques. It aims to assist in scenarios where you have limited training data or want to enhance the quality and diversity of your existing dataset.

## 🎯 Key Use Cases

1. **Augment Existing Training Data**: Expand your existing training data with additional synthetic examples generated by Gretel Navigator. This can help improve the robustness and generalization of your AI models.

2. **Create Diverse Training or Evaluation Data**: Generate diverse training or evaluation data from plain text or seed examples. This ensures your AI models are exposed to a wide range of scenarios and edge cases during training.

3. **Address Data Limitations**: Generate additional examples to fill gaps in your dataset, particularly for underrepresented classes, rare events, or challenging scenarios. This helps improve your model's ability to handle diverse real-world situations.

4. **Mitigate Bias and Toxicity**: Generate training examples that are unbiased and non-toxic by incorporating diverse perspectives and adhering to ethical guidelines. This promotes fairness and responsible AI development.

5. **Enhance Model Performance**: Improve the performance of your AI models across various tasks by training them on diverse synthetic data generated by Gretel Navigator.

## 🔧 Getting Started

To start using Gretel Navigator, you'll need:

1. A Gretel account (free accounts are available).
2. Seed text or input/output pairs to create or augment AI training data.

## 📂 Input Data Formats

Gretel Navigator supports the following formats for input data:

- Existing AI training or evaluation data formats:
  - Input/Output pair format (or instruction/response) with any number of ground truth or "context fields".
  - Plain text data.
- File formats:
  - Hugging Face dataset
  - CSV
  - JSON
  - JSONL

## 📤 Output

Gretel Navigator generates one additional training example per row in the input/output pair format. You can specify requirements for the input and output pairs in the configuration. Run the process multiple times to scale your data to any desired level.

## 🌟 AI Alignment Techniques

Gretel Navigator incorporates AI alignment techniques to generate high-quality synthetic data:

- Diverse Instruction and Response Generation
- AI-Aligning-AI Methodology (AAA) for iterative data quality enhancement
- Quality Evaluation
- Bias and Toxicity Detection

By leveraging these techniques, Gretel Navigator helps you create training data that leads to more robust, unbiased, and high-performing AI models.

---

Ready to enhance your AI training data and unlock the full potential of your models? Let's get started with Gretel Navigator! 🚀
"""


def main():
    st.set_page_config(page_title="Gretel", layout="wide")
    st.title("🎨 Gretel Navigator: Enhance Your AI Training Data")

    with st.expander("Introduction", expanded=False):
        st.markdown(WELCOME_MARKDOWN)

    st.subheader("Step 1: API Key Validation")
    with st.expander("API Key Configuration", expanded=True):
        api_key = st.text_input(
            "Enter your Gretel API key (https://console.gretel.ai)",
            value="",
            type="password",
            help="Your Gretel API key for authentication",
        )
        if "gretel" not in st.session_state:
            st.session_state.gretel = None
        if st.button("Validate API Key"):
            if api_key:
                try:
                    st.session_state.gretel = Gretel(api_key=api_key, validate=True)
                    st.success("API key validated. Connection successful!")
                except Exception as e:
                    st.error(f"Error connecting to Gretel: {str(e)}")
            else:
                st.warning("Please enter your Gretel API key to proceed.")
        if st.session_state.gretel is None:
            st.stop()

    st.subheader("Step 2: Data Source Selection")
    with st.expander("Data Source", expanded=True):
        data_source = st.radio(
            "Select data source",
            options=[
                "Upload a file",
                "Select a dataset from Hugging Face",
                "Use a sample dataset",
            ],
            help="Choose whether to upload a file, select a dataset from Hugging Face, or use a sample dataset",
        )

        df = None
        if data_source == "Upload a file":
            uploaded_file = st.file_uploader(
                "Upload a CSV, JSON, or JSONL file",
                type=["csv", "json", "jsonl"],
                help="Upload the dataset file in CSV, JSON, or JSONL format",
            )

            if uploaded_file is not None:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".json"):
                    df = pd.read_json(uploaded_file)
                elif uploaded_file.name.endswith(".jsonl"):
                    df = pd.read_json(uploaded_file, lines=True)
                st.success(f"File uploaded successfully: {uploaded_file.name}")

        elif data_source == "Select a dataset from Hugging Face":
            huggingface_dataset = st.text_input(
                "Hugging Face Dataset Repository",
                help="Enter the name of the Hugging Face dataset repository (e.g., 'squad')",
            )

            huggingface_split = st.selectbox(
                "Dataset Split",
                options=["train", "validation", "test"],
                help="Select the dataset split to use",
            )

            if st.button("Load Hugging Face Dataset"):
                if huggingface_dataset:
                    try:
                        with st.spinner("Loading dataset from Hugging Face..."):
                            dataset = load_dataset(
                                huggingface_dataset, split=huggingface_split
                            )
                            df = dataset.to_pandas()
                        st.success(
                            f"Dataset loaded from Hugging Face repository: {huggingface_dataset}"
                        )
                    except Exception as e:
                        st.error(f"Error loading dataset from Hugging Face: {str(e)}")
                else:
                    st.warning("Please provide a Hugging Face dataset repository name.")

        elif data_source == "Use a sample dataset":
            st.write("Try a sample dataset to get started quickly.")
            if st.button("Try Sample Dataset"):
                try:
                    df = pd.read_csv(SAMPLE_DATASET_URL)
                    st.success("Sample dataset loaded successfully.")
                except Exception as e:
                    st.error(f"Error downloading sample dataset: {str(e)}")

        if df is not None:
            st.session_state.df = df
            st.session_state.selected_fields = list(df.columns)
            st.write(
                f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns."
            )
        else:
            df = st.session_state.get("df")

    st.subheader("Step 3: Data Preview and Configuration")
    if df is not None:
        with st.expander("Data Preview", expanded=True):
            st.dataframe(df.head())

        with st.expander("Input Fields Selection", expanded=True):
            st.write(
                "Select the context fields to provide the LLM access to for generating input/output pairs. This can include existing instructions and responses. All selected fields will be treated as ground truth data."
            )

            selected_fields = []
            for column in df.columns:
                if st.checkbox(
                    column,
                    value=column in st.session_state.get("selected_fields", []),
                    key=f"checkbox_{column}",
                ):
                    selected_fields.append(column)

            st.session_state.selected_fields = selected_fields

        with st.expander("Advanced Options", expanded=False):

            output_instruction_field = st.text_input(
                "Synthetic instruction field",
                value=st.session_state.get("output_instruction_field", "instruction"),
                help="Specify the name of the output field for generated instructions",
            )
            st.session_state.output_instruction_field = output_instruction_field

            output_response_field = st.text_input(
                "Synthetic response field",
                value=st.session_state.get("output_response_field", "response"),
                help="Specify the name of the output field for generated responses",
            )
            st.session_state.output_response_field = output_response_field

            num_records = st.number_input(
                "Max number of records from input data to process",
                min_value=1,
                max_value=len(df),
                value=len(df),
                help="Specify the number of records to process",
            )
            st.session_state.num_records = num_records

            num_instructions = st.number_input(
                "Number of diverse candidate instructions",
                min_value=1,
                value=st.session_state.get("num_instructions", 5),
                help="Specify the number of instructions to generate",
            )
            st.session_state.num_instructions = num_instructions

            num_responses = st.number_input(
                "Number of diverse candidateresponses",
                min_value=1,
                value=st.session_state.get("num_responses", 5),
                help="Specify the number of responses to generate",
            )
            st.session_state.num_responses = num_responses

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("temperature", 0.8),
                step=0.1,
                help="Adjust the temperature for response generation",
            )
            st.session_state.temperature = temperature

            max_tokens_instruction = st.slider(
                "Max tokens (instruction)",
                min_value=1,
                max_value=1024,
                value=st.session_state.get("max_tokens_instruction", 100),
                help="Specify the maximum number of tokens for instructions",
            )
            st.session_state.max_tokens_instruction = max_tokens_instruction

            max_tokens_response = st.slider(
                "Max tokens (response)",
                min_value=1,
                max_value=1024,
                value=st.session_state.get("max_tokens_response", 100),
                help="Specify the maximum number of tokens for responses",
            )
            st.session_state.max_tokens_response = max_tokens_response

        with st.expander("Model Configuration", expanded=True):
            st.markdown("### Primary Navigator Models")

            tabular_models = st.session_state.gretel.factories.get_navigator_model_list(
                "tabular"
            )
            navigator_tabular = st.selectbox(
                "Navigator Tabular",
                options=tabular_models,
                index=st.session_state.get("navigator_tabular_index", 0),
                help="Select the primary Navigator tabular model",
            )
            st.session_state.navigator_tabular_index = tabular_models.index(
                navigator_tabular
            )

            nl_models = st.session_state.gretel.factories.get_navigator_model_list(
                "natural_language"
            )
            navigator_llm = st.selectbox(
                "Navigator LLM",
                options=nl_models,
                index=st.session_state.get("navigator_llm_index", 0),
                help="Select the primary Navigator LLM",
            )
            st.session_state.navigator_llm_index = nl_models.index(navigator_llm)

            st.markdown("---")
            st.markdown("### AI Align AI (AAA)")
            st.write(
                "AI Align AI (AAA) is a technique that iteratively improves the quality and coherence of generated outputs by using multiple LLMs for co-teaching and self-teaching. Enabling AAA will enhance the overall quality of the synthetic data, but it may slow down the generation process."
            )

            use_aaa = st.checkbox(
                "Use AI Align AI (AAA)",
                value=st.session_state.get("use_aaa", True),
                help="Enable or disable the use of AI Align AI.",
            )
            st.session_state.use_aaa = use_aaa

            co_teach_llms = []  # Initialize co_teach_llms with an empty list

            if use_aaa:
                st.markdown("#### Navigator Co-teaching LLMs")
                st.write(
                    "Select additional Navigator LLMs for co-teaching in AAA. It is recommended to use different LLMs than the primary Navigator LLM for this step."
                )

                for model in nl_models:
                    if model != navigator_llm:
                        if st.checkbox(model, value=True, key=f"checkbox_{model}"):
                            co_teach_llms.append(model)
                    else:
                        if st.checkbox(model, value=False, key=f"checkbox_{model}"):
                            co_teach_llms.append(model)
                st.session_state.co_teach_llms = co_teach_llms

            st.markdown("---")
            st.markdown("### Format Prompts")

            instruction_format_prompt = st.text_area(
                "Instruction Format Prompt",
                value=st.session_state.get(
                    "instruction_format_prompt",
                    "A well-formulated question or command in everyday English.",
                ),
                help="Specify the format prompt for instructions",
            )
            st.session_state.instruction_format_prompt = instruction_format_prompt

            response_format_prompt = st.text_area(
                "Response Format Prompt",
                value=st.session_state.get(
                    "response_format_prompt",
                    "A well-formulated response to the question in everyday English.",
                ),
                help="Specify the format prompt for responses",
            )
            st.session_state.response_format_prompt = response_format_prompt

        with st.expander("Download SDK Code", expanded=False):
            config_text = f"""
        # Create the data augmentation configuration
        config = DataAugmentationConfig(
            input_fields={st.session_state.selected_fields},
            output_instruction_field="{output_instruction_field}",
            output_response_field="{output_response_field}",
            num_instructions={num_instructions},
            num_responses={num_responses},
            temperature={temperature},
            max_tokens_instruction={max_tokens_instruction},
            max_tokens_response={max_tokens_response},
            api_key=YOUR_GRETEL_API_KEY,
            navigator_tabular="{navigator_tabular}",
            navigator_llm="{navigator_llm}",
            co_teach_llms={co_teach_llms},
            instruction_format_prompt="{instruction_format_prompt}",
            response_format_prompt="{response_format_prompt}"
        )

        # Create the data augmenter and perform augmentation
        augmenter = DataAugmenter(
            df,
            config,
            use_aaa={use_aaa},
            output_file="results.csv",
            verbose=True,
        )
        new_df = augmenter.augment()
        """
            st.code(config_text, language="python")
            st.download_button(
                label="Download SDK Code",
                data=config_text,
                file_name="data_augmentation_code.py",
                mime="text/plain",
            )

        start_stop_container = st.empty()

        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("🚀 Start")
        with col2:
            stop_button = st.button("🛑 Stop")

        if start_button:
            with st.expander("Augmentation Results", expanded=True):
                st.subheader("Augmentation Results")
                progress_bar = st.progress(0)
                tab1, tab2 = st.tabs(["Augmented Data", "Logs"])
                with tab1:
                    augmented_data_placeholder = st.empty()
                    st.info(
                        "Click on the 'Logs' tab to see and debug real-time logging for each record as it is generated by the agents."
                    )
                with tab2:
                    log_container = st.empty()
                    logs = []
                    max_log_lines = 50

                def custom_log_handler(msg):
                    nonlocal logs
                    logs.append(msg)
                    if len(logs) > max_log_lines:
                        logs = logs[-max_log_lines:]
                    log_text = "\n".join(logs)
                    log_container.text(log_text)

                handler = StreamlitLogHandler(custom_log_handler)
                logger = logging.getLogger("navigator_helpers")
                logger.addHandler(handler)
                config = DataAugmentationConfig(
                    input_fields=selected_fields,
                    output_instruction_field=output_instruction_field,
                    output_response_field=output_response_field,
                    num_instructions=num_instructions,
                    num_responses=num_responses,
                    temperature=temperature,
                    max_tokens_instruction=max_tokens_instruction,
                    max_tokens_response=max_tokens_response,
                    api_key=api_key,
                    navigator_tabular=navigator_tabular,
                    navigator_llm=navigator_llm,
                    co_teach_llms=co_teach_llms,
                    instruction_format_prompt=instruction_format_prompt,
                    response_format_prompt=response_format_prompt,
                )
                augmented_data = []
                start_time = time.time()
                with st.spinner("Generating synthetic data..."):
                    for index in range(num_records):
                        row = df.iloc[index]
                        augmenter = DataAugmenter(
                            pd.DataFrame([row]),
                            config,
                            use_aaa=use_aaa,
                            output_file="results.csv",
                            verbose=True,
                        )
                        new_df = augmenter.augment()
                        augmented_data.append(new_df)
                        augmented_data_placeholder.subheader("Augmented Data")
                        augmented_data_placeholder.dataframe(
                            pd.concat(augmented_data, ignore_index=True)
                        )
                        progress = (index + 1) / num_records
                        progress_bar.progress(progress)

                        elapsed_time = time.time() - start_time
                        records_processed = index + 1
                        records_remaining = num_records - records_processed
                        est_time_per_record = (
                            elapsed_time / records_processed
                            if records_processed > 0
                            else 0
                        )
                        est_time_remaining = est_time_per_record * records_remaining

                        progress_text = f"Progress: {progress:.2%} | Records Processed: {records_processed} | Records Remaining: {records_remaining} | Est. Time per Record: {est_time_per_record:.2f}s | Est. Time Remaining: {est_time_remaining:.2f}s"
                        progress_bar.text(progress_text)

                        time.sleep(0.1)
                logger.removeHandler(handler)
                st.success("Data augmentation completed!")
        if stop_button:
            st.warning("Augmentation stopped by the user.")
            st.stop()
        else:
            st.info(
                "Please upload a file or select a dataset from Hugging Face to proceed."
            )


if __name__ == "__main__":
    main()