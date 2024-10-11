import json
import logging
import queue
import random
import threading
import time

import pandas as pd
import streamlit as st
import yaml

from gretel_client import Gretel

from navigator_helpers.config_generator import ConfigGenerator
from navigator_helpers.data_models import DataModel
from navigator_helpers.synthetic_data_generator import SyntheticDataGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if "gretel" not in st.session_state:
    st.session_state.gretel = None
if "config" not in st.session_state:
    st.session_state.config = None
if "generated_data" not in st.session_state:
    st.session_state.generated_data = None
if "current_step" not in st.session_state:
    st.session_state.current_step = 1
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = False
if "data_generation_started" not in st.session_state:
    st.session_state.data_generation_started = False
if "data_generation_process_started" not in st.session_state:
    st.session_state.data_generation_process_started = False
if "run_id" not in st.session_state:
    st.session_state.run_id = None


# Load the external default prompts file
def load_default_prompts(file_path="streamlit_prompts.yaml"):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


# Initialize default prompts from external file
default_prompts = load_default_prompts()


def render_pretty_json(data):
    # Prepare a list to collect formatted output
    output_lines = []

    # Function to recursively format JSON or dict values
    def format_json(key, value):
        output_lines.append(f"{key}")
        output_lines.append("----------------------")
        if isinstance(value, dict) or isinstance(value, list):
            # Recursively handle nested dictionaries and lists
            output_lines.append("'''")
            if isinstance(value, dict):
                for k, v in value.items():
                    format_json(k, v)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    output_lines.append(f"- {item}")
            output_lines.append("'''")
        else:
            # Handle simple key-value pairs
            output_lines.append("'''")
            output_lines.append(str(value))
            output_lines.append("'''")

    # Iterate through the top-level keys in the data
    if isinstance(data, dict):
        for key, value in data.items():
            format_json(key, value)

    # Join the collected lines into a single string and display it in st.code
    formatted_output = "\n".join(output_lines)
    st.code(formatted_output, language="text")


def reset_data_generation_state():
    if "data_generation_thread" in st.session_state:
        thread = st.session_state.data_generation_thread
        if thread is not None and thread.is_alive():
            st.session_state.stop_event.set()
            thread.join()
            st.session_state.stop_event.clear()
    keys_to_reset = [
        "data_generation_thread",
        "generated_data",
        "data_queue",
        "log_queue",
        "log_messages",
        "data_generation_in_progress",
        "generated_count",
        "total_records",
        "stop_event",
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


def main():
    st.set_page_config(page_title="Gretel Synthetic Data Assistant", layout="wide")
    st.title("Gretel Synthetic Data Assistant")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    steps = [
        "1. Authentication",
        "2. Dataset Definition",
        "3. Synthetic Data Generation",
    ]
    choice = st.sidebar.radio("Go to", steps, index=st.session_state.current_step - 1)
    st.session_state.current_step = steps.index(choice) + 1

    if st.session_state.current_step == 1:
        authentication_page()
    elif st.session_state.current_step == 2:
        data_configuration_page()
    elif st.session_state.current_step == 3:
        data_generation_page()


def authentication_page():
    st.header("Step 1: Authenticate with API Key")

    # Input for API key
    api_key = st.text_input(
        "Enter your [Gretel API key](https://console.gretel.ai/users/me/key) 🔑",
        type="password",
    )

    if st.button("Validate API Key"):
        if api_key:
            with st.spinner("Validating API key..."):
                try:
                    st.session_state.gretel = Gretel(api_key=api_key, validate=True)
                    st.success("API key validated. Connection successful!")
                    st.session_state.current_step = 2
                    st.rerun()
                except Exception as e:
                    st.error(f"Error connecting to Gretel: {str(e)}")
        else:
            st.warning("Please enter your Gretel API key to proceed.")


def data_configuration_page():
    st.header("Step 2: Define your Dataset")
    if st.session_state.gretel is None:
        st.warning("Please authenticate first.")
        st.session_state.current_step = 1
        st.rerun()
        return

    st.markdown(
        """
    In this step, we'll transform your description into a comprehensive data model. 
    Our AI expands on your input, considering various data types, relationships, and edge cases 
    to create a diverse and scalable dataset structure.
    """
    )

    st.markdown("#### Choose a Prompt (Optional)")

    prompts_per_row = 5
    prompt_names = list(default_prompts.keys())

    # Create rows of buttons
    for i in range(0, len(prompt_names), prompts_per_row):
        cols = st.columns(prompts_per_row)
        for col, name in zip(cols, prompt_names[i : i + prompts_per_row]):
            if col.button(name):
                st.session_state.user_task = default_prompts[name]

    # Dataset definition form
    with st.form(key="data_config_form"):
        st.subheader("Describe Your Dataset")

        # Display the text area and populate with the selected prompt if any
        user_task = st.text_area(
            "",
            value=st.session_state.get("user_task", ""),
            placeholder="Describe the dataset you want to create...",
            height=400,
        )

        # Define the diversity options and their mappings
        diversity_options = {
            "Low": {"target": 100, "max_tags": 2},
            "Medium": {"target": 10000, "max_tags": 4},
            "High": {"target": 100000, "max_tags": 6},
            "Very High": {"target": 1000000, "max_tags": 8},
        }

        # Use select_slider to choose the diversity level
        diversity_level = st.select_slider(
            "Select diversity target",
            options=["Low", "Medium", "High", "Very High"],
            value="Medium",
            help=(
                "Select the diversity level for your dataset:\n"
                "- **Low**: Suitable for a few thousand examples (< 1,000 records)\n"
                "- **Medium**: Suitable for smaller datasets (10,000 records)\n"
                "- **High**: Generates a larger, more diverse dataset (100,000 records)\n"
                "- **Very High**: Generates a very large and highly diverse dataset (1,000,000 records)"
            ),
        )

        diversity_target = diversity_options[diversity_level]["target"]
        max_tags = diversity_options[diversity_level]["max_tags"]

        submit_button = st.form_submit_button(label="✨ Generate Dataset Definition")

    # Map the selected level to the numeric value and max tags
    diversity_target = diversity_options[diversity_level]["target"]
    max_tags = diversity_options[diversity_level]["max_tags"]

    if submit_button:
        reset_data_generation_state()

        with st.spinner("Generating synthetic dataset definition..."):
            # Reset records in case new config is generated
            st.session_state.generated_data = pd.DataFrame()

            config_generator = ConfigGenerator(
                api_key="prompt", model="gretelai/gpt-auto"
            )
            config_generator.set_user_task(user_task)
            tags = config_generator.generate_tags(diversity_target=diversity_target)
            config_generator.set_tags(tags.tags[:max_tags])

            config_generator.generate_data_model()

            # Trim the tags based on the diversity level
            st.session_state.config = config_generator.get_config()

    if st.session_state.config:
        st.subheader("Generated Gretel Dataset Definition")

        with st.expander("🔍 What just happened?"):
            st.markdown(
                """
            - We analyzed your description and identified key data elements.
            - Our AI expanded this into a full data model, including:
              - Appropriate data types for each field
              - Realistic value ranges and distributions
              - Relationships between different data points
              - Edge cases and rare scenarios for more robust data
            - This model is now ready to generate millions of diverse, realistic records.
            """
            )

        # Use an expander for editing the configuration
        with st.expander("View/Edit Definition", expanded=True):
            if st.session_state.edit_mode:
                edited_config = st.text_area(
                    "Edit the YAML configuration", st.session_state.config, height=300
                )
                col1, col2 = st.columns(2)
                with col1:
                    update_clicked = st.button("Update Definition")
                with col2:
                    cancel_clicked = st.button("Cancel Edit")
                if update_clicked:
                    try:
                        yaml.safe_load(edited_config)  # Validate YAML
                        st.session_state.config = edited_config
                        st.success("Dataset Definition updated!")
                        st.session_state.edit_mode = False
                        st.rerun()
                    except yaml.YAMLError as e:
                        st.error(f"Invalid YAML: {str(e)}")
                if cancel_clicked:
                    st.session_state.edit_mode = False
                    st.rerun()
            else:
                st.code(st.session_state.config, language="yaml")
                edit_clicked = st.button("Edit Definition")
                if edit_clicked:
                    st.session_state.edit_mode = True
                    st.rerun()

        # Download button for configuration
        st.download_button(
            label="⬇️ Download Dataset Definition",
            data=st.session_state.config,
            file_name="config.yaml",
            mime="text/yaml",
        )

        if st.button("️⏩ Data Generation"):
            st.session_state.current_step = 3
            st.rerun()


def data_generation_page():
    # Adjust cell wrapping with CSS to ensure all cell content is visible
    st.markdown(
        """
        <style>
        .stTable th, .stTable td {
            white-space: normal;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("Step 3: Synthetic Data Generation")

    if st.session_state.gretel is None or st.session_state.config is None:
        st.warning("Please complete authentication and data configuration first.")
        st.session_state.current_step = 1
        st.rerun()
        return

    # Initialize session state variables
    if "data_generation_thread" not in st.session_state:
        st.session_state.data_generation_thread = None
    if (
        "generated_data" not in st.session_state
        or st.session_state.generated_data is None
    ):
        st.session_state.generated_data = pd.DataFrame()
    if "data_queue" not in st.session_state:
        st.session_state.data_queue = queue.Queue()
    if "log_queue" not in st.session_state:
        st.session_state.log_queue = queue.Queue()
    if "log_messages" not in st.session_state:
        st.session_state.log_messages = []
    if "data_generation_in_progress" not in st.session_state:
        st.session_state.data_generation_in_progress = False
    if "generated_count" not in st.session_state:
        st.session_state.generated_count = 0
    if "total_records" not in st.session_state:
        st.session_state.total_records = 0
    if "stop_event" not in st.session_state:
        st.session_state.stop_event = threading.Event()

    def run_data_generation(model_definition, data_queue, log_queue, stop_event):
        try:
            # Custom logging handler to capture logs
            class QueueHandler(logging.Handler):
                def emit(self, record):
                    log_entry = self.format(record)
                    log_queue.put(log_entry)

            # Set up logging for the generator
            logger = logging.getLogger("SyntheticDataGenerator")
            logger.setLevel(logging.INFO)
            logger.handlers = []  # Remove any existing handlers
            queue_handler = QueueHandler()
            queue_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            logger.addHandler(queue_handler)

            # Pass the logger to SyntheticDataGenerator
            generator = SyntheticDataGenerator(model_definition, logger=logger)

            data_gen = generator.generate_data()

            for record in data_gen:
                if stop_event.is_set():
                    logger.info("Stop event set, terminating data generation.")
                    break  # Exit the loop to stop the generation

                data_queue.put(record)

            data_queue.put("DONE")
            logger.info("Data generation process completed or stopped.")

        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            data_queue.put(error_msg)
            log_queue.put(error_msg)
            logger.error(f"An error occurred: {error_msg}")

        finally:
            # Ensure resources are cleaned up when the thread is finishing
            logger.info("Cleaning up the data generation thread.")
            log_queue.put("Thread has stopped.")

    if st.button("️✨ Start") and not st.session_state.data_generation_in_progress:
        st.session_state.data_generation_in_progress = True
        st.session_state.generated_data = pd.DataFrame()
        st.session_state.generated_count = 0
        st.session_state.data_queue = queue.Queue()
        st.session_state.log_queue = queue.Queue()
        st.session_state.stop_event.clear()  # Reset the stop event

        # Load the YAML configuration
        config_content = st.session_state.config
        model_definition = DataModel.from_yaml(config_content)

        # Total records to generate
        st.session_state.total_records = model_definition.num_examples

        # Start the data generation thread
        st.session_state.data_generation_thread = threading.Thread(
            target=run_data_generation,
            args=(
                model_definition,
                st.session_state.data_queue,
                st.session_state.log_queue,
                st.session_state.stop_event,
            ),
            daemon=True,
        )
        st.session_state.data_generation_thread.start()
        st.rerun()

    if st.session_state.data_generation_in_progress:

        with st.expander("🔍 What's happening in this step?"):
            st.markdown(
                """
            This assistant has constructed a high-quality synthetic data generation pipeline for creating advanced AI training data. Key features include:

            * **Self-Thinking Approach**: Extended reasoning time for more realistic and nuanced data.
            * **Compound AI System**: Multi-stage process refines and optimizes data quality.
            * **Evolutionary Approach**: Data evolves over generations, capturing complex relationships.
            * **Hallucination Reduction**: Combats AI hallucinations for more accurate data.
            * **Contextual Generation**: Uses your tags to create relevant, diverse, domain-specific data at scale.
            * **Continuous Validation**: Ensures each example meets your specified criteria.

            This pipeline generates synthetic datasets that are statistically similar to real data while capturing intricate patterns and domain-specific nuances crucial for robust AI model training.
            """
            )

        # Placeholders for dynamic updates
        data_placeholder = st.empty()
        latest_record_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        progress_text = st.empty()
        stop_button_placeholder = st.empty()
        log_placeholder = st.empty()

        # Create a placeholder for stopping the generation
        if stop_button_placeholder.button("⏹️ Stop"):
            st.session_state.stop_event.set()  # Signal the thread to stop
            status_placeholder.warning("Stopping data generation...")

        # Download button for configuration
        st.download_button(
            label="⬇️ Download Dataset Definition",
            data=st.session_state.config,
            file_name="config.yaml",
            mime="text/yaml",
        )

        # Retrieve data from the queue
        new_records = []
        while not st.session_state.data_queue.empty():
            item = st.session_state.data_queue.get()
            if item == "DONE":
                st.session_state.data_generation_in_progress = False
                break
            elif isinstance(item, str) and item.startswith("ERROR:"):
                st.error(item)
                st.session_state.data_generation_in_progress = False
                break
            else:
                new_records.append(item)

        if new_records:
            new_data = pd.DataFrame(new_records)
            st.session_state.generated_data = pd.concat(
                [st.session_state.generated_data, new_data], ignore_index=True
            )
            st.session_state.generated_count += len(new_records)

        # Update status and progress
        total_records = st.session_state.total_records
        generated_count = st.session_state.generated_count
        if total_records > 0:
            progress = min(int((generated_count / total_records) * 100), 100)
        else:
            progress = 0

        if st.session_state.stop_event.is_set():
            status_placeholder.warning("Stopping data generation...")
        else:
            status_placeholder.info("Data generation in progress...")
        progress_bar.progress(progress)
        progress_text.text(
            f"Generated {generated_count} out of {total_records} records ({progress}%)."
        )

        # Update data preview
        if not st.session_state.generated_data.empty:
            data_placeholder.subheader("Generated Data Preview")
            data_placeholder.dataframe(st.session_state.generated_data)

            # Display the latest record using st.code for clean formatting
            latest_record = st.session_state.generated_data.iloc[-1].to_dict()
            latest_record_placeholder.subheader("Latest Record")
            render_pretty_json(latest_record)

        # Log display
        st.subheader("Logs")
        if st.session_state.log_messages:
            # Display the latest log message
            st.write(st.session_state.log_messages[-1])

        # Use an expander for full logs to keep the interface clean
        with st.expander("View Full Logs"):
            # Retrieve log messages from the queue
            while not st.session_state.log_queue.empty():
                log_message = st.session_state.log_queue.get()
                if log_message:
                    st.session_state.log_messages.append(log_message)

            # Keep only the last 100 log messages
            st.session_state.log_messages = st.session_state.log_messages[-100:]

            # Display log messages in a scrollable text area
            log_text = "\n".join(st.session_state.log_messages)
            if not log_text:
                log_text = "No logs available."
            st.text_area("Full logs", log_text, height=200, key="log_display")

        # Check if the thread has finished
        if not st.session_state.data_generation_thread.is_alive():
            st.session_state.data_generation_in_progress = False
            if st.session_state.stop_event.is_set():
                st.success("Data generation stopped successfully.")
            else:
                st.success("Data generation completed successfully.")
        else:
            time.sleep(1)
            st.rerun()

        # Add more detailed information about the generation process
        with st.expander("🔬 Generation Details"):
            st.code(st.session_state.config, language="yaml")

        # Add a "Did You Know?" section
        if random.random() < 0.2:  # Show this 20% of the time
            st.info(
                random.choice(
                    [
                        "Did you know? Our system can generate complex database schemas that mimic real-world production environments.",
                        "Fun fact: The evolutionary approach we use can create more diverse and realistic data than traditional methods.",
                        "Interesting tidbit: Our system can generate both SQL queries and the natural language prompts that might lead to those queries!",
                    ]
                )
            )

    elif not st.session_state.generated_data.empty:
        st.subheader("Generated Data Preview")
        st.dataframe(st.session_state.generated_data)

        # Display the latest record below
        latest_record = st.session_state.generated_data.iloc[-1].to_dict()
        st.subheader("Latest Record")
        # Use st.code() with syntax highlighting
        latest_record_json = json.dumps(latest_record, indent=4)
        st.code(latest_record_json, language="json")

        # Download button for the generated data
        csv = st.session_state.generated_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="generated_data.csv",
            mime="text/csv",
        )

        # Add a section highlighting the value of the generated data
        st.subheader("🌟 Value of Your Generated Data")
        st.markdown(
            """
        - **High Quality**: Evolved over multiple generations for accuracy and relevance.
        - **Domain-Specific**: Tailored to your chosen domain and complexity level.
        - **Diverse**: Includes a wide range of scenarios, including edge cases.
        - **Realistic**: Mimics real-world data patterns and relationships.
        - **Hallucination-Free**: Rigorously validated to minimize AI-generated errors.
        """
        )

    if st.button("Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


if __name__ == "__main__":
    main()
