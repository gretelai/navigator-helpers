import logging
import time
from io import StringIO

import pandas as pd
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


def main():
    st.set_page_config(page_title="Navigator Training Data Augmentation", layout="wide")
    st.title("👋 Welcome to Navigator Self Align")

    intro_expander = st.expander("Introduction", expanded=True)

    with intro_expander:
        st.markdown(
            """Navigator Self Align is an interface for creating high quality, diverse training data examples via synthetic data generation. It leverages techniques such as diverse instruction and response generation, and an iterative AI alignment process including AI-Aligning-AI methodology (AAA) to iterative enhance data quality. The system integrates multiple LLMs in a co-teaching and self-improvement process, generating diverse and high-quality synthetic instructions, responses, and leverages evaluations for quality, adherence, toxicity, and bias.
        """
        )

    st.subheader("Step 1: API Key Validation")
    with st.expander("API Key Configuration", expanded=True):
        api_key = st.text_input(
            "Enter your Gretel API key",
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
            options=["Upload a file", "Select a dataset from Hugging Face"],
            help="Choose whether to upload a file or select a dataset from Hugging Face",
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

        if df is not None:
            st.session_state.df = df
            st.session_state.selected_fields = list(df.columns)
        else:
            df = st.session_state.get("df")

    st.subheader("Step 3: Data Preview and Configuration")
    if df is not None:
        with st.expander("Data Preview", expanded=True):
            st.dataframe(df.head())

        with st.expander("Input Fields Selection", expanded=True):
            selected_fields = []
            for column in df.columns:
                if st.checkbox(column, key=f"checkbox_{column}"):
                    selected_fields.append(column)

            st.session_state.selected_fields = selected_fields

        with st.expander("Output Fields Configuration", expanded=True):
            output_instruction_field = st.text_input(
                "Output instruction field",
                value=st.session_state.get(
                    "output_instruction_field", "generated_instruction"
                ),
                help="Specify the name of the output field for generated instructions",
            )
            st.session_state.output_instruction_field = output_instruction_field

            output_response_field = st.text_input(
                "Output response field",
                value=st.session_state.get(
                    "output_response_field", "generated_response"
                ),
                help="Specify the name of the output field for generated responses",
            )
            st.session_state.output_response_field = output_response_field

            num_records = st.number_input(
                "Number of records to process",
                min_value=1,
                max_value=len(df),
                value=len(df),
                help="Specify the number of records to process",
            )
            st.session_state.num_records = num_records

        with st.expander("Configuration Options", expanded=True):
            num_instructions = st.number_input(
                "Number of instructions",
                min_value=1,
                value=st.session_state.get("num_instructions", 5),
                help="Specify the number of instructions to generate",
            )
            st.session_state.num_instructions = num_instructions

            num_responses = st.number_input(
                "Number of responses",
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

            max_tokens_instruction = st.number_input(
                "Max tokens (instruction)",
                min_value=1,
                value=st.session_state.get("max_tokens_instruction", 100),
                help="Specify the maximum number of tokens for instructions",
            )
            st.session_state.max_tokens_instruction = max_tokens_instruction

            max_tokens_response = st.number_input(
                "Max tokens (response)",
                min_value=1,
                value=st.session_state.get("max_tokens_response", 150),
                help="Specify the maximum number of tokens for responses",
            )
            st.session_state.max_tokens_response = max_tokens_response

            use_aaa = st.checkbox(
                "Use AI Align AI (AAA)",
                value=st.session_state.get("use_aaa", True),
                help="Enable or disable the use of AI Align AI",
            )
            st.session_state.use_aaa = use_aaa

        with st.expander("Model Configuration", expanded=True):
            tabular_models = st.session_state.gretel.factories.get_navigator_model_list(
                "tabular"
            )
            navigator_tabular = st.selectbox(
                "Navigator Tabular",
                options=tabular_models,
                index=st.session_state.get("navigator_tabular_index", 0),
                help="Select the tabular model to use for navigation",
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
                help="Select the language model to use for navigation",
            )
            st.session_state.navigator_llm_index = nl_models.index(navigator_llm)

            co_teach_llms = []
            for model in nl_models:
                if model != navigator_llm:
                    if st.checkbox(model, value=True, key=f"checkbox_{model}"):
                        co_teach_llms.append(model)
                else:
                    if st.checkbox(model, value=False, key=f"checkbox_{model}"):
                        co_teach_llms.append(model)

            st.session_state.co_teach_llms = co_teach_llms

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

        start_stop_container = st.empty()

        col1, col2 = st.columns(2)
        with col1:
            start_button = st.button("🚀 Start")
        with col2:
            stop_button = st.button("Stop")

        if start_button:
            intro_expander.empty()
            with st.expander("Augmentation Results", expanded=True):
                st.subheader("Augmentation Results")

                progress_bar = st.progress(0)

                tab1, tab2 = st.tabs(["Logs", "Augmented Data"])

                with tab1:
                    log_container = st.empty()
                    logs = []

                with tab2:
                    augmented_data_placeholder = st.empty()

                def custom_log_handler(msg):
                    nonlocal logs
                    # logs.insert(0, msg)
                    logs.append(msg)
                    log_text = "\n".join(logs)
                    log_container.text(log_text)

                    if (
                        "🆕 Starting the process of generating a new augmented record."
                        in msg
                    ):
                        time.sleep(3)
                        logs.clear()

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

                    progress_bar.progress((index + 1) / num_records)
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
