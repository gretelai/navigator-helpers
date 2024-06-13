import streamlit as st
import pandas as pd
import logging
import uuid
from io import StringIO

from gretel_client import Gretel
from navigator_helpers import DataAugmentationConfig, DataAugmenter, StreamlitLogHandler

# Default configuration values
NAVIGATOR_TABULAR = "gretelai/auto"
NAVIGATOR_LLM = "gretelai/gpt-auto"
CO_TEACH_LLMS = [
    "gretelai/gpt-llama3-8b",
    "gretelai/gpt-mistral7b",
]  # List of co-teaching models

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

    st.title("Navigator Training Data Augmentation")

    # API key input
    api_key = st.text_input(
        "Enter your Gretel API key",
        value="",
        type="password",
    )

    if "gretel" not in st.session_state:
        st.session_state.gretel = None

    if st.button("Validate"):
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

    # File upload
    uploaded_file = None

    if st.session_state.gretel is not None:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a CSV or JSON file", type=["csv", "json"]
        )

        if uploaded_file is not None:
            # Read the uploaded file into a DataFrame
            df = (
                pd.read_csv(uploaded_file)
                if uploaded_file.name.endswith(".csv")
                else pd.read_json(uploaded_file)
            )

            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Auto-select columns based on names
            default_context = [col for col in df.columns if "context" in col]
            default_instruction = next(
                (col for col in df.columns if "instruction" in col), None
            )
            default_response = next((col for col in df.columns if "response" in col), None)

            # Column selection
            context_fields = st.multiselect(
                "Select context fields", df.columns, default=default_context
            )
            instruction_field = st.selectbox(
                "Select instruction field",
                df.columns,
                index=df.columns.get_loc(default_instruction) if default_instruction else 0,
            )
            response_field = st.selectbox(
                "Select response field",
                df.columns,
                index=df.columns.get_loc(default_response) if default_response else 0,
            )

            # Record selection
            num_records = st.number_input(
                "Number of records to process",
                min_value=1,
                max_value=len(df),
                value=len(df),
            )

            # Configuration options
            st.subheader("Configuration Options")
            col1, col2 = st.columns(2)
            with col1:
                num_instructions = st.number_input(
                    "Number of instructions", min_value=1, value=5
                )
                num_responses = st.number_input("Number of responses", min_value=1, value=5)
                temperature = st.slider(
                    "Temperature", min_value=0.0, max_value=1.0, value=0.8, step=0.1
                )
            with col2:
                max_tokens_instruction = st.number_input(
                    "Max tokens (instruction)", min_value=1, value=100
                )
                max_tokens_response = st.number_input(
                    "Max tokens (response)", min_value=1, value=150
                )
                use_aaa = st.checkbox(
                    "Use AI Align AI (AAA)", value=False
                )  # Set AAA off by default

            # New configuration options
            st.subheader("Model Configuration")

            # Load tabular models
            tabular_models = st.session_state.gretel.factories.get_navigator_model_list(
                "tabular"
            )
            navigator_tabular = st.selectbox(
                "Navigator Tabular", options=tabular_models, index=0
            )

            # Load natural language models
            nl_models = st.session_state.gretel.factories.get_navigator_model_list(
                "natural_language"
            )
            navigator_llm = st.selectbox("Navigator LLM", options=nl_models, index=0)

            # Select co-teach models
            co_teach_llms = st.multiselect(
                "Co-teaching LLMs",
                options=nl_models,
                default=nl_models[1:3] if len(nl_models) >= 3 else [],
            )

            instruction_format_prompt = st.text_area(
                "Instruction Format Prompt",
                value="A well-formulated question or command in everyday English.",
            )
            response_format_prompt = st.text_area(
                "Response Format Prompt",
                value="A well-formulated response to the question in everyday English.",
            )

            # Set a flag indicating that the configuration options have been presented
            st.session_state.config_presented = True

        # Display the "Start Augmentation" button only if the configuration options have been presented
        if st.session_state.get("config_presented", False):
            if st.button("Start Augmentation"):
                # Create a new screen or area for augmentation results
                augmentation_screen = st.empty()

                with augmentation_screen.container():
                    st.subheader("Augmentation Results")

                    # Create a progress bar
                    progress_bar = st.progress(0)

                    # Create tabs for logs and augmented data
                    tab1, tab2 = st.tabs(["Logs", "Augmented Data"])
                    logs = ""
                    log_entries = []

                    with tab1:
                        # Placeholder for logging output
                        log_text_area = st.empty()
                        log_text_area.text_area("Logs", height=600)

                    with tab2:
                        # Placeholder for augmented data
                        augmented_data_placeholder = st.empty()

                    # Custom log handler that appends logs to the text area
                    text_area_id = "logs_text_area"

                    scroll_script = f"""
                    <script>
                    var textArea = document.getElementById("{text_area_id}");
                    textArea.scrollTop = textArea.scrollHeight;
                    </script>
                    """

                    # Custom log handler that appends logs to the text area
                    def custom_log_handler(msg):
                        nonlocal logs
                        logs += msg + "\n"

                        # Generate a unique ID for each log message
                        log_id = str(uuid.uuid4())

                        # Define the JavaScript code to scroll the text area to the bottom
                        scroll_script = f"""
                        <script>
                        var textArea = document.getElementById("{log_id}");
                        textArea.scrollTop = textArea.scrollHeight;
                        </script>
                        """

                        # Update the log text area with the new log message and unique ID
                        log_text_area.text_area("Logs", value=logs, height=600, key=log_id)

                        # Include the script to scroll the text area to the bottom
                        st.markdown(scroll_script, unsafe_allow_html=True)

                    # Create a custom log handler and attach it to the logger
                    handler = StreamlitLogHandler(custom_log_handler)
                    logger = logging.getLogger("navigator_helpers")
                    logger.addHandler(handler)

                    # Data augmentation configuration
                    config = DataAugmentationConfig(
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

                    for field in context_fields:
                        config.add_field(field, field_type="context")
                    config.add_field(instruction_field, field_type="instruction")
                    config.add_field(response_field, field_type="response")

                    augmented_data = []

                    # Process data row by row
                    for index in range(num_records):
                        # Get a single row from the DataFrame
                        row = df.iloc[index]

                        # Initialize the data augmenter for the current row
                        augmenter = DataAugmenter(
                            pd.DataFrame([row]),
                            config,
                            use_aaa=use_aaa,
                            use_examples=True,
                            output_file="results.csv",
                            verbose=True,
                        )

                        # Perform data augmentation for the current row
                        new_df = augmenter.augment()

                        # Append the augmented data to the list
                        augmented_data.append(new_df)

                        # Update the output data table
                        augmented_data_placeholder.subheader("Augmented Data")
                        augmented_data_placeholder.dataframe(
                            pd.concat(augmented_data, ignore_index=True)
                        )

                        # Update the progress bar
                        progress_bar.progress((index + 1) / num_records)

                        # Sleep for a short duration to allow the UI to update
                        import time

                        time.sleep(0.1)

                        # Clear the log entries for the next iteration
                        log_entries.clear()

                    # Remove the logging handler
                    logger.removeHandler(handler)

                    st.success("Data augmentation completed!")


if __name__ == "__main__":
    main()
