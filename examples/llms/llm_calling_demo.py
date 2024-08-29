import argparse
import logging

from argparse import Namespace
from pathlib import Path

from autogen import AssistantAgent, UserProxyAgent

from navigator_helpers.data_synthesis import logger
from navigator_helpers.llms import init_llms, LLMConfig, LLMWrapper, str_to_message
from navigator_helpers.llms.autogen import AutogenAdapter
from navigator_helpers.logs import configure_logger


def main(config_path: str) -> None:
    llm_registry = init_llms(Path(config_path), fail_on_error=False)

    # Filter configs by tag
    llm_configs = llm_registry.find_by_tags({"base"})

    # Use it with autogen
    run_autogen_example(llm_configs)

    # Use it directly
    run_direct_example(llm_configs)


def run_autogen_example(llm_configs: list[LLMConfig]) -> None:
    adapter = AutogenAdapter(llm_configs)
    assistant = adapter.initialize_agent(
        AssistantAgent("assistant", llm_config=adapter.create_llm_config())
    )
    user_proxy = UserProxyAgent(
        "user_proxy", human_input_mode="NEVER", code_execution_config=False
    )
    result = user_proxy.initiate_chat(
        assistant,
        message="Write python code to print 'Hello World!'",
    )
    logger.info(f"Autogen result: {result}")


def run_direct_example(llm_configs: list[LLMConfig]) -> None:
    llm = LLMWrapper.from_llm_configs(llm_configs)
    result = llm.completion(
        [
            str_to_message("You are an experienced Python developer.", role="system"),
            str_to_message("Write python code to print 'Hello World!'"),
        ]
    )
    logger.info(f"Direct LLM call result: {result}")


def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(description="Run demo.")
    parser.add_argument(
        "--config", required=True, type=str, help="Path to the configuration file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    configure_logger(level=logging.INFO)
    args = parse_arguments()
    main(args.config)
