import json
import logging
import os
import random
import re
import string
import time

from dataclasses import dataclass
from io import StringIO
from typing import Any, Dict, List, Optional, Union

import autogen
import pandas as pd

from autogen.io import base, console

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Custom logging filter
class NoParentFilter(logging.Filter):
    def filter(self, record):
        return not record.name.startswith(("HTTP Request"))


# Apply the filter to the root logger
logging.getLogger().addFilter(NoParentFilter())


@dataclass
class EvolutionConfig:
    population_size: int = 2
    num_generations: int = 3
    mutation_rate: float = 0.6
    temperature: float = 0.7
    verbose: bool = False
    primary_model_tags: List[str] = ("llama-3.1-8b",)


@dataclass
class AAAConfig:
    temperature: float = 0.7
    num_co_teachers: int = 2
    co_teach_temperature: float = 0.5
    num_suggestions: int = 3
    verbose: bool = False
    primary_model_tags: List[str] = ("llama-3.1-8b",)
    co_teaching_model_tags: List[str] = ("llama-3.1-8b",)


class BaseTextProcessor:
    """
    Base class for text processing operations.

    This class provides common functionality for text processing,
    including setting up agents and user proxies.
    """

    def __init__(self, config: Union["EvolutionConfig", "AAAConfig"]) -> None:
        self.config = config
        self.agents = self._setup_agents()
        self.user_proxy = self._setup_user_proxy()
        self._setup_logging()

    def _setup_logging(self) -> None:
        """
        Set up logging configuration to suppress unwanted logs.
        """
        # Suppress logs from specific modules
        logging.getLogger("groq").setLevel(logging.WARNING)
        logging.getLogger("mistralai").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

    def _setup_agents(self) -> Dict[str, autogen.AssistantAgent]:
        """
        Set up the autogen agents for text processing.

        Returns:
            Dict[str, autogen.AssistantAgent]: A dictionary of agent names to agent objects.

        Raises:
            ValueError: If no primary model configuration is found.
        """

        DEFAULT_DATA_SYSTEM_MESSAGE = """You are an AI assistant specializing in critically evaluating data,
        particularly tabular data used for training machine learning models.
        Your primary functions include identifying errors, discrepancies, and opportunities for data augmentation.
        You leverage expertise across various domains, including data analysis, data engineering, and a general understanding of programming concepts.
        
        Your Capabilities:
        - Data Analysis: You can examine datasets to identify patterns, anomalies, and potential issues.
        - Error Detection: You're adept at spotting inconsistencies, outliers, and data quality issues.
        - Data Augmentation Suggestions: You can propose methods to enrich or expand datasets to improve their utility for machine learning.
        - Domain Knowledge Application: You can apply relevant domain expertise to contextualize data and identify domain-specific issues or opportunities.
        - Statistical Understanding: You can perform and interpret basic statistical analyses to support your evaluations.
        """

        LLM_CONFIG_LIST = json.loads(os.environ.get("LLM_CONFIG_LIST"))
        PRIMARY_CONFIG_LIST_TO_USE = autogen.filter_config(
            LLM_CONFIG_LIST, {"tags": self.config.primary_model_tags}
        )

        agents = {}
        # Setup primary agent
        if PRIMARY_CONFIG_LIST_TO_USE:
            primary_model_config = PRIMARY_CONFIG_LIST_TO_USE[0]
            primary_model_name = primary_model_config.get("model", "unknown")
            primary_api_type = primary_model_config.get("api_type", "unknown")
            primary_agent_name = f"PrimaryAgent_{primary_api_type}-{primary_model_name}"

            primary_llm_config = dict(primary_model_config)
            primary_llm_config.update(
                {
                    "temperature": self.config.temperature,
                    "cache_seed": None,  # Disable caching
                }
            )

            agents["primary"] = autogen.AssistantAgent(
                name=primary_agent_name,
                llm_config=primary_llm_config,
                system_message=DEFAULT_DATA_SYSTEM_MESSAGE,
            )
        else:
            raise ValueError("No primary model configuration found.")

        # Setup co-teaching agents only if co_teaching_model_tags are provided
        if hasattr(self.config, "co_teaching_model_tags"):
            CO_TEACHING_CONFIG_LIST_TO_USE = autogen.filter_config(
                LLM_CONFIG_LIST, {"tags": self.config.co_teaching_model_tags}
            )
            for i in range(self.config.num_co_teachers):
                model_config = CO_TEACHING_CONFIG_LIST_TO_USE[
                    i % len(CO_TEACHING_CONFIG_LIST_TO_USE)
                ]
                model_name = model_config.get("model", "unknown")
                api_type = model_config.get("api_type", "unknown")
                agent_name = f"CoTeachAgent_{i}_{api_type}-{model_name}"

                llm_config = dict(model_config)
                llm_config.update(
                    {
                        "temperature": self.config.co_teach_temperature,
                        "cache_seed": None,  # Disable caching
                    }
                )

                agents[f"co_teach_{i}"] = autogen.AssistantAgent(
                    name=agent_name,
                    llm_config=llm_config,
                    system_message=DEFAULT_DATA_SYSTEM_MESSAGE,
                )

        return agents

    def _setup_user_proxy(self) -> autogen.UserProxyAgent:
        """
        Set up the user proxy agent.

        Returns:
            autogen.UserProxyAgent: The user proxy agent.
        """
        return autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            code_execution_config=False,  # don't allow code execution
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda x: True,  # terminate after one response
        )

    def _get_agent_response(self, agent: autogen.AssistantAgent, prompt: str) -> str:
        """
        Get a response from an agent for a given prompt.

        Args:
            agent (autogen.AssistantAgent): The agent to get a response from.
            prompt (str): The prompt to send to the agent.

        Returns:
            str: The agent's response.
        """
        # Add a unique identifier to the prompt
        unique_id = "".join(random.choices(string.ascii_letters + string.digits, k=10))
        timestamp = time.time()
        full_prompt = f"[Request ID: {unique_id}, Timestamp: {timestamp}]\n{prompt}"

        # Clear the chat history of the user proxy
        self.user_proxy.reset()

        # Initiate a chat with the agent
        self.user_proxy.initiate_chat(agent, message=full_prompt)

        # Retrieve the last message from the conversation, which should be the agent's response
        conversation = self.user_proxy.chat_messages[agent]
        if len(conversation) >= 2:  # Ensure there's a response
            return conversation[-1]["content"]
        else:
            return ""

    def parse_jsonl(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse JSONL text, handling potential newlines within JSON objects.

        Args:
            text (str): The JSONL text to parse.

        Returns:
            List[Dict[str, Any]]: A list of parsed JSON objects.
        """
        parsed_objects: List[Dict[str, Any]] = []
        try:
            # This pattern matches JSON objects, allowing for nested structures and newlines
            json_pattern = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}")
            json_objects = json_pattern.findall(text)
            for obj in json_objects:
                try:
                    parsed_objects.append(json.loads(obj))
                except json.JSONDecodeError:
                    logger.info(f"Warning: Could not parse JSON object: {obj[:100]}...")
        except Exception as e:
            logger.info(f"An error occurred during parsing: {str(e)}")
        return parsed_objects

    def is_jsonl(self, text: str) -> bool:
        """
        Check if the given text is in JSONL format.

        Args:
            text (str): The text to check.

        Returns:
            bool: True if the text is in JSONL format, False otherwise.
        """
        parsed = self.parse_jsonl(text)
        return len(parsed) > 0 and all(isinstance(obj, dict) for obj in parsed)

    def extract_json(self, text: str) -> str:
        """
        Extract JSON content from the given text.

        Args:
            text (str): The text to extract JSON from.

        Returns:
            str: The extracted JSON content, or an empty JSON string if no valid JSON is found.
        """
        # Remove the string TERMINATE if it's present at the end of the text
        text = re.sub(r"\s*TERMINATE\s*$", "", text)

        # Find the start of the JSON content
        json_start = re.search(r"```(?:json)?\s*\{", text)
        if json_start:
            # Remove everything before the start of the JSON content
            text = text[json_start.start() :]

            # Remove the opening code block marker
            text = re.sub(r"^```(?:json)?\s*", "", text)

            # Find and remove the closing code block marker
            text = re.sub(r"\s*```\s*$", "", text)

        # Try to parse the text as JSON
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            # If parsing fails, try to find JSON-like content
            json_pattern = r"\{(?:[^{}]|\{[^{}]*\})*\}"
            matches = re.findall(json_pattern, text, re.DOTALL)

            # Sort matches by length and try to parse the largest one
            matches.sort(key=len, reverse=True)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    return json.dumps(parsed, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    continue

        # If no valid JSON found, return an empty JSON string
        return "{}"


class EvolutionTextProcessor(BaseTextProcessor):
    """
    Text processor that uses an evolutionary algorithm to improve text.
    """

    def __init__(self, evolution_config: EvolutionConfig):
        super().__init__(evolution_config)

    def _convert_to_jsonl(self, df: pd.DataFrame) -> str:
        return df.to_json(orient="records", lines=True)

    def _convert_to_dataframe(self, jsonl_str: str) -> pd.DataFrame:
        return pd.read_json(StringIO(jsonl_str), lines=True)

    def evolve(
        self,
        initial_text: Union[str, pd.DataFrame],
        context: Optional[Union[str, pd.DataFrame]] = None,
    ) -> Union[str, pd.DataFrame]:
        """
        Evolve the initial text using an evolutionary algorithm.

        Args:
            initial_text (Union[str, pd.DataFrame]): The initial text to evolve.
            context (Optional[Union[str, pd.DataFrame]]): Optional context for evolution.

        Returns:
            Union[str, pd.DataFrame]: The evolved text.
        """

        evolution_start_time = time.time()

        # Track input types
        initial_text_is_df = isinstance(initial_text, pd.DataFrame)
        context_is_df = (
            isinstance(context, pd.DataFrame) if context is not None else False
        )

        # Convert to JSONL only if it's a DataFrame
        initial_text_jsonl = (
            self._convert_to_jsonl(initial_text) if initial_text_is_df else initial_text
        )
        context_jsonl = self._convert_to_jsonl(context) if context_is_df else context

        # Clear all agent conversation histories
        for agent in self.agents.values():
            self.user_proxy.reset()

        if self.config.verbose:
            logger.info("🧬 🧬 🧬 Starting the Evolution process ...")

        # Create initial population
        current_population = self._create_initial_population(
            initial_text_jsonl, context_jsonl
        )
        for gen in range(self.config.num_generations):
            if self.config.verbose:
                logger.info(f"Generation {gen + 1}/{self.config.num_generations}")

            mutated_texts = self._mutate_population(current_population, context_jsonl)
            all_texts = current_population + mutated_texts
            ranked_texts = self._rank_texts(all_texts, context_jsonl)
            current_population = ranked_texts[: self.config.population_size]

            if self.config.verbose:
                logger.info(
                    f"Best text after generation {gen + 1}: {current_population[0][:100]}"
                )

        evolved_text = current_population[0]

        evolution_end_time = time.time()
        evolution_duration = evolution_end_time - evolution_start_time

        if self.config.verbose:
            logger.info(f"-----------------")
            logger.info(
                f"Evolution process completed in {evolution_duration:.2f} seconds"
            )

        # Convert back to DataFrame if the input was a DataFrame
        if initial_text_is_df:
            evolved_text = self._convert_to_dataframe(evolved_text)

        return evolved_text

    def _create_initial_population(
        self, initial_text: str, context: Optional[str] = None
    ) -> List[str]:
        population = [initial_text]
        if self.config.verbose:
            logger.info(
                f"🌱 Creating initial population of size {self.config.population_size}"
            )
        while len(population) < self.config.population_size:
            population.append(self._mutate_text(initial_text, context))
        return population

    def _mutate_text(self, text: str, context: Optional[str] = None) -> str:
        if self.config.verbose:
            logger.info(f"Mutating text: {text[:100]}")

        is_jsonl_input = self.is_jsonl(text)

        if is_jsonl_input:
            jsonl_objects = self.parse_jsonl(text)
            mutation_prompt = f"""
            Mutate each line of the ORIGINAL_JSONL while maintaining its structure and intent.
            The mutation should introduce some variation or improvement to the text.
            {'Take into account context in JSONL_CONTEXT. DO NOT include JSONL_CONTEXT itself.' if context else ''}
            DO NOT modify schema of ORIGINAL_JSONL. DO NOT respond with code to do the mutating.

            {'JSONL_CONTEXT:' if context else ''} {context}

            ORIGINAL_JSONL:
            {json.dumps(jsonl_objects, indent=2)}

            Respond with a JSON object in this JSONL format (do not include ellipses, which are for demonstration only):
            {{
                "mutated_jsonl": [
                    {{
                        "mutated": {{mutated line 1 JSON object}}
                    }},
                    {{
                        "mutated": {{mutated line 2 JSON object}}
                    }},
                    ...
                    {{
                        "mutated": {{mutated line {len(jsonl_objects)} JSON object}}
                    }}
                ],
                "mutation_explanation": "A brief explanation of the changes made"
            }}

            Provide only the JSON object, with no additional text.
            """
        else:
            mutation_prompt = f"""
            Mutate the following ORIGINAL_TEXT while maintaining its intent.
            The mutation should introduce some variation or improvement to the text.
            {'Take into account context in CONTEXT. DO NOT include CONTEXT itself.' if context else ''}
            Do not respond with code to do the mutating.

            {'CONTEXT:' if context else ''} {context}

            ORIGINAL_TEXT: {text}

            Respond with a JSON object in the following format:
            {{
                "mutated_text": "The mutated version of the text",
                "mutation_explanation": "A brief explanation of the changes made"
            }}

            Provide only the JSON object, with no additional text.
            """

        response = self._get_agent_response(self.agents["primary"], mutation_prompt)
        response = self.extract_json(response)

        try:
            mutation_data = json.loads(response)
            if is_jsonl_input:
                mutated_jsonl = mutation_data.get("mutated_jsonl", [])
                mutated_text = "\n".join(
                    json.dumps(item["mutated"]) for item in mutated_jsonl
                )
            else:
                mutated_text = mutation_data.get("mutated_text")

            if mutated_text:
                if self.config.verbose:
                    logger.info(f"Mutation result: {mutated_text[:100]}")
                return mutated_text
            else:
                logger.info(
                    "Mutation failed to produce a valid result. Returning original text."
                )
                return text

        except json.JSONDecodeError:
            logger.info(
                "Error parsing JSON response in mutation. Returning original text."
            )
            return text
        except Exception as e:
            logger.info(
                f"Unexpected error in mutation: {str(e)}. Returning original text."
            )
        return text

    def _mutate_population(
        self, population: List[str], context: Optional[str] = None
    ) -> List[str]:
        return [
            (
                self._mutate_text(text, context)
                if random.random() < self.config.mutation_rate
                else text
            )
            for text in population
        ]

    def _rank_texts(self, texts: List[str], context: Optional[str] = None) -> List[str]:
        if len(texts) < 2:
            return texts  # If there's only one text, return it without ranking

        if self.config.verbose:
            logger.info(f"🔢 Ranking original and mutated texts")

        evaluation_prompt = f"""
        Rank the following texts based on their quality, relevance {' to the CONTEXT' if context else ''}, and complexity.
        Do not respond with code to do the ranking

        {'CONTEXT:' if context else ''} {context}

        Texts:
        {json.dumps({i: text for i, text in enumerate(texts)})}

        Provide the ranking as a JSON object with the following structure:
        {{
            "ranking": [
                {{"position": 1, "text_index": 0}},
                {{"position": 2, "text_index": 1}},
            ]
        }}

        Provide only the JSON object, with no additional text.
        """

        ranking_response = self._get_agent_response(
            self.agents["primary"], evaluation_prompt
        )
        ranking_response = self.extract_json(ranking_response)

        return self._parse_ranking(ranking_response, texts)

    def _parse_ranking(
        self, ranking_response: str, original_texts: List[str]
    ) -> List[str]:
        try:
            ranking_data = json.loads(ranking_response)
            ranked_texts = []
            for item in ranking_data["ranking"]:
                text_index = item["text_index"]
                if 0 <= text_index < len(original_texts):
                    ranked_texts.append(original_texts[text_index])

            # Add any missing texts to the end of the list
            for text in original_texts:
                if text not in ranked_texts:
                    ranked_texts.append(text)

            return ranked_texts
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.info(
                f"Error parsing ranking response: {e}. Returning original order."
            )
            return original_texts


class AAATextProcessor(BaseTextProcessor):
    """
    Text processor that uses the AI-Align-AI (AAA) method to improve text.
    """

    def __init__(self, aaa_config: AAAConfig):
        super().__init__(aaa_config)

    def _convert_to_jsonl(self, df: pd.DataFrame) -> str:
        return df.to_json(orient="records", lines=True)

    def _convert_to_dataframe(self, jsonl_str: str) -> pd.DataFrame:
        return pd.read_json(StringIO(jsonl_str), lines=True)

    def apply_aaa(
        self,
        text: Union[str, pd.DataFrame],
        context: Optional[Union[str, pd.DataFrame]] = None,
    ) -> Union[str, pd.DataFrame]:
        """
        Apply the AAA method to improve the input text.

        Args:
            text (Union[str, pd.DataFrame]): The text to improve.
            context (Optional[Union[str, pd.DataFrame]]): Optional context for improvement.

        Returns:
            Union[str, pd.DataFrame]: The improved text.
        """

        aaa_start_time = time.time()

        # Track input types
        text_is_df = isinstance(text, pd.DataFrame)
        context_is_df = (
            isinstance(context, pd.DataFrame) if context is not None else False
        )

        # Convert to JSONL only if it's a DataFrame
        text_jsonl = self._convert_to_jsonl(text) if text_is_df else text
        context_jsonl = self._convert_to_jsonl(context) if context_is_df else context

        # Clear all agent conversation histories
        for agent in self.agents.values():
            self.user_proxy.reset()

        if self.config.verbose:
            logger.info("🦾 🦿 🤖 Starting the AAA process ...")

        co_teaching_results, co_teaching_explanations = self._co_teach(
            text_jsonl, context_jsonl
        )
        best_co_teaching_text, best_co_teaching_explanation = (
            self._select_best_co_teaching(
                text_jsonl, co_teaching_results, co_teaching_explanations
            )
        )
        suggestions = self._generate_suggestions(best_co_teaching_text, context_jsonl)
        self_teaching_text, self_teaching_explanation = self._self_teach(
            best_co_teaching_text, suggestions, context_jsonl
        )
        final_text = self._make_final_selection(
            text_jsonl,
            best_co_teaching_text,
            best_co_teaching_explanation,
            self_teaching_text,
            self_teaching_explanation,
            context_jsonl,
        )

        aaa_end_time = time.time()
        aaa_duration = aaa_end_time - aaa_start_time

        if self.config.verbose:
            logger.info(f"-----------------")
            logger.info(f"AAA process completed in {aaa_duration:.2f} seconds")

        # Convert back to DataFrame if the input was a DataFrame
        if text_is_df:
            final_text = self._convert_to_dataframe(final_text)

        return final_text

    def _co_teach(self, text: str, context: Optional[str] = None) -> List[str]:
        if self.config.verbose:
            logger.info("👩‍🏫 Starting co-teaching process")

        is_jsonl_input = self.is_jsonl(text)

        if is_jsonl_input:
            jsonl_objects = self.parse_jsonl(text)
            co_teaching_prompt = f"""
            Improve ORIGINAL_JSONL {' based on JSONL_CONTEXT. DO NOT include JSONL_CONTEXT itself.' if context else '.'}
            DO NOT modify schema of ORIGINAL_JSONL. DO NOT respond with code to do the improving.

            Improvements should address inconsistencies, inaccuracies, mistakes, and highlight opportunities to improve quality.

            {'JSONL_CONTEXT:' if context else ''} {context}

            ORIGINAL_JSONL:
            {json.dumps(jsonl_objects, indent=2)}

            Respond with a JSON object in the following format:
            {{
                "improved_jsonl": [
                    {{
                        improved line 1 JSON object
                    }},
                    {{
                        improved line 2 JSON object
                    }},
                    ...
                    {{
                        improved line {len(jsonl_objects)} JSON object
                    }},

                ],
                "improvement_explanation": "A brief explanation of the changes made"
            }}

            Provide only the JSON object, with no additional text.
            """
        else:
            co_teaching_prompt = f"""
            Improve the following ORIGINAL_TEXT while maintaining its intent.
            Do not respond with code to do the improving.
            Improvements should address inconsistencies, inaccuracies, mistakes, and highlight opportunities to improve quality.
            {'Consider the provided CONTEXT in your improvement.' if context else ''}

            {'CONTEXT:' if context else ''} {context}

            ORIGINAL_TEXT: {text}

            Respond with a JSON object in the following format:
            {{
                "improved_text": "The improved version of the text",
                "improvement_explanation": "A brief explanation of the changes made"
            }}

            Provide only the JSON object, with no additional text.
            """

        co_teaching_results = []
        co_teaching_explanations = []
        for agent in self.agents.values():
            if agent.name.startswith("CoTeach"):
                response = self._get_agent_response(agent, co_teaching_prompt)
                response = self.extract_json(response)
                try:
                    improvement_data = json.loads(response)

                    if is_jsonl_input:
                        improved_jsonl = improvement_data.get("improved_jsonl", [])
                        improved_text = "\n".join(
                            json.dumps(item, ensure_ascii=False)
                            for item in improved_jsonl
                        )

                        # TODO: we could have explanations per line here
                        improvement_explanation = improvement_data.get(
                            "improvement_explanation", ""
                        )
                    else:
                        improved_text = improvement_data.get("improved_text")
                        improvement_explanation = improvement_data.get(
                            "improvement_explanation", ""
                        )

                    if improved_text:
                        co_teaching_results.append(improved_text)
                    else:
                        logger.info(
                            f"Co-teaching with {agent.name} failed to produce a valid result. Using original text."
                        )
                        co_teaching_results.append(text)

                    if improvement_explanation:
                        co_teaching_explanations.append(improvement_explanation)
                    else:
                        logger.info(
                            f"Co-teaching with {agent.name} failed to produce a valid improvement explanation."
                        )
                        co_teaching_results.append("")

                except json.JSONDecodeError:
                    logger.info(
                        f"Error parsing JSON response from {agent.name} in co-teaching. Using original text."
                    )
                    co_teaching_results.append(text)

        if self.config.verbose:
            logger.info(
                f"Co-teaching process completed. Generated {len(co_teaching_results)} results."
            )

        return co_teaching_results, co_teaching_explanations

    def _select_best_co_teaching(
        self,
        original_text: str,
        co_teaching_results: List[str],
        co_teaching_explanations: List[str],
    ) -> str:
        if self.config.verbose:
            logger.info("🔍 Selecting best co-teaching result")

        is_jsonl_input = self.is_jsonl(original_text)
        # Remove duplicates while preserving order
        unique_results = list(dict.fromkeys(co_teaching_results))

        # If all results are identical, return the first one
        if len(unique_results) == 1:
            return unique_results[0]

        # If there are no unique results (all were identical to the original), return the original
        if not unique_results:
            return original_text

        if is_jsonl_input:
            selection_prompt = f"""
            Select the best improvement of ORIGINAL_JSONL from the IMPROVEMENT_OPTIONS below.
            Take IMPROVEMENT_EXPLANATIONS into account. Do not respond with code to do the selecting.

            ORIGINAL_JSONL: {original_text}

            IMPROVEMENT_OPTIONS:
            {json.dumps({f"option_{i+1}": text for i, text in enumerate(unique_results)}, indent=2)}

            IMPROVEMENT_EXPLANATIONS: {co_teaching_explanations}

            Respond with a JSON object in the following format:
            {{
                "selected_option": "option_X",
                "reasoning": "Your explanation for this choice"
            }}

            Provide only the JSON object, with no additional text.
            """
        else:
            selection_prompt = f"""
            Select the best improvement of ORIGINAL_TEXT from the IMPROVEMENT_OPTIONS below:
            Take IMPROVEMENT_EXPLANATIONS into account. Do not respond with code to do the selecting.

            ORIGINAL_TEXT: {original_text}

            IMPROVEMENT_OPTIONS:
            {json.dumps({f"option_{i+1}": text for i, text in enumerate(unique_results)}, indent=2)}

            IMPROVEMENT_EXPLANATIONS: {co_teaching_explanations}

            Respond with a JSON object in the following format:
            {{
                "selected_option": "option_X",
                "reasoning": "Your explanation for this choice"
            }}

            Provide only the JSON object, with no additional text.
            """

        selection_response = self._get_agent_response(
            self.agents["primary"], selection_prompt
        )
        selection_response = self.extract_json(selection_response)

        try:
            selection_data = json.loads(selection_response)
            selected_option = selection_data.get("selected_option")

            if selected_option and selected_option.startswith("option_"):
                option_index = int(selected_option.split("_")[1]) - 1
                if 0 <= option_index < len(unique_results):
                    selected_text = unique_results[option_index]
                    selected_reasoning = selection_data.get("reasoning", "")
                    if self.config.verbose:
                        logger.info(f"Selected best co-teaching result.")
                    return selected_text, selected_reasoning

            # If parsing fails or selection is invalid, return the first unique result
            selected_text = unique_results[0]
            if self.config.verbose:
                logger.info(f"Invalid selection. Defaulting to first unique result.")
            return selected_text, ""

        except json.JSONDecodeError:
            logger.info("Error parsing JSON response. Returning first unique result.")
            selected_text = unique_results[0]
            if self.config.verbose:
                logger.info(f"Selected best co-teaching result (default).")
            return selected_text, ""

    def _generate_suggestions(
        self, text: str, context: Optional[str] = None
    ) -> List[str]:
        if self.config.verbose:
            logger.info(f"Generating {self.config.num_suggestions} suggestions...")

        is_jsonl_input = self.is_jsonl(text)

        if is_jsonl_input:
            jsonl_objects = self.parse_jsonl(text)
            suggestions_prompt = f"""
            For each JSON object in ORIGINAL_JSONL, provide exactly {self.config.num_suggestions}
            suggestions to further improve it{' based on JSONL_CONTEXT. DO NOT include JSONL_CONTEXT itself.' if context else '.'}

            Suggestions should use natural language and address inconsistencies, inaccuracies, mistakes, and highlight opportunities to improve quality.

            {'JSONL_CONTEXT:' if context else ''} {context}

            ORIGINAL_JSONL:
            {json.dumps(jsonl_objects, indent=2)}

            Respond with a JSON object in this format (do not include ellipses, which are for demonstration only):
            {{
                "suggestions": [
                    List of {self.config.num_suggestions} plain english suggestions for line 1,
                    List of {self.config.num_suggestions} plain english suggestions for line 2,
                    ...
                    List of {self.config.num_suggestions} plain english suggestions for line {len(jsonl_objects)},
                ]
            }}

            Ensure you provide exactly {self.config.num_suggestions} suggestions.
            Provide only the JSON object, with no additional text.
            """
        else:
            suggestions_prompt = f"""
            Provide exactly {self.config.num_suggestions} suggestions to further improve
            ORIGINAL_TEXT{' based on CONTEXT.' if context else '.'}
            Do not respond with code to do the suggesting.
            Suggestions should address inconsistencies, inaccuracies, mistakes, and highlight opportunities to improve quality.

            {'CONTEXT:' if context else ''} {context}

            ORIGINAL_TEXT: {text}

            Respond with a JSON object in the following format:
            {{
                "suggestions": [
                    "Suggestion 1",
                    "Suggestion 2",
                ]
            }}

            Ensure you provide exactly {self.config.num_suggestions} suggestions.
            Provide only the JSON object, with no additional text.
            """

        response = self._get_agent_response(self.agents["primary"], suggestions_prompt)
        response = self.extract_json(response)

        try:
            suggestions_data = json.loads(response)
            suggestions = suggestions_data.get("suggestions", [])

            # Ensure we have exactly the requested number of suggestions
            if len(suggestions) > self.config.num_suggestions:
                suggestions = suggestions[: self.config.num_suggestions]
            while len(suggestions) < self.config.num_suggestions:
                suggestions.append(
                    "No additional suggestion" if not is_jsonl_input else "{}"
                )

            if self.config.verbose:
                logger.info("Suggestions generated.")

            return suggestions

        except json.JSONDecodeError:
            logger.info("Error parsing JSON response. Returning default suggestions.")
            default_suggestion = "{}" if is_jsonl_input else "No additional suggestion"
            return [default_suggestion] * self.config.num_suggestions

    def _self_teach(
        self, text: str, suggestions: List[str], context: Optional[str] = None
    ) -> str:
        if self.config.verbose:
            logger.info("Starting self-teaching process...")

        is_jsonl_input = self.is_jsonl(text)

        if is_jsonl_input:
            jsonl_objects = self.parse_jsonl(text)
            self_teaching_prompt = f"""
            Improve text in ORIGINAL_JSONL by incorporating the SUGGESTIONS below
            {'and taking into account context in JSONL_CONTEXT. DO NOT include JSONL_CONTEXT itself.' if context else '.'}
            DO NOT modify schema of JSONL. DO NOT respond with code to do the suggesting.

            {'JSONL_CONTEXT:' if context else ''} {context}

            ORIGINAL_JSONL:
            {text}

            SUGGESTIONS:
            {suggestions}

            Respond with a JSON object in this JSONL format (do not include ellipses, which are for demonstration only):
            {{
                "improved_jsonl": [
                    {{
                        improved line 1 JSONL object
                    }},
                    {{
                        improved line 2 JSONL object
                    }},
                    ...
                    {{
                        improved line {len(jsonl_objects)} JSONL object
                    }},
                ],
                "changes_made": "A brief explanation of the changes made"
            }}

            Provide only the JSON object, with no additional text.
            """
        else:
            self_teaching_prompt = f"""
            Improve the ORIGINAL_TEXT by incorporating the SUGGESTIONS below
            {'and taking into account context in CONTEXT. DO NOT include CONTEXT itself.' if context else '.'}
            Do not respond with code to do the incorporating.

            {'CONTEXT:' if context else ''} {context}

            ORIGINAL_TEXT: {text}

            SUGGESTIONS:
            {json.dumps(suggestions)}

            Respond with a JSON object in the following format:
            {{
                "improved_text": "The improved version of the text",
                "changes_made": "A brief explanation of the changes made"
            }}

            Provide only the JSON object, with no additional text.
            """

        response = self._get_agent_response(
            self.agents["primary"], self_teaching_prompt
        )
        response = self.extract_json(response)

        try:
            improvement_data = json.loads(response)
            if is_jsonl_input:
                improved_jsonl = improvement_data.get("improved_jsonl", [])
                improved_text = "\n".join(
                    json.dumps(item, ensure_ascii=False) for item in improved_jsonl
                )

                # TODO: we could have explanations per line here
                improvement_explanation = improvement_data.get("changes_made", "")
            else:
                improved_text = improvement_data.get("improved_text", text)
                improvement_explanation = improvement_data.get("changes_made", "")

            if self.config.verbose:
                logger.info(f"Self-teaching completed.")

            return improved_text, improvement_explanation

        except json.JSONDecodeError:
            logger.info("Error parsing JSON response. Returning original text.")
            return text

    def _make_final_selection(
        self,
        original: str,
        co_teaching: str,
        co_teaching_expl: str,
        self_teaching: str,
        self_teaching_expl: str,
        context: Optional[str] = None,
    ) -> str:
        if self.config.verbose:
            logger.info("Making final selection...")

        final_selection_prompt = f"""
        Evaluate and select the best version from the Original/Co-Teaching/Self-Teaching options below.
        Take Co-Teaching-Explanation and Self-Teaching-Explanation into account.
        Evaluate each option based on clarity, completeness, and overall quality.
        Choose the option that best addresses the original intent while including all
        relevant information {'and taking into account context in CONTEXT.' if context else '.'}
        Do not respond with code to do the selecting.

        {'CONTEXT:' if context else ''} {context}

        Original: {original}

        Co-Teaching: {co_teaching}

        Self-Teaching: {self_teaching}

        Co-Teaching-Explanation: {co_teaching_expl}

        Self-Teaching-Explanation: {self_teaching_expl}

        Respond with a JSON object in the following format:
        {{
            "selected_option": "Original" or "Co-Teaching" or "Self-Teaching",
            "reasoning": "Your explanation for this choice"
        }}

        Provide only the JSON object, with no additional text:
        """

        response = self._get_agent_response(
            self.agents["primary"], final_selection_prompt
        )
        response = self.extract_json(response)

        try:
            selection_data = json.loads(response)
            selected_option = selection_data.get("selected_option")

            if selected_option == "Original":
                selected_text = original
            elif selected_option == "Co-Teaching":
                selected_text = co_teaching
            elif selected_option == "Self-Teaching":
                selected_text = self_teaching
            else:
                # If the selection is invalid, default to the original text
                logger.info(
                    f"Invalid selection '{selected_option}'. Defaulting to original text."
                )
                selected_text = original

            if self.config.verbose:
                logger.info(f"Final selection made: {selected_option}")

            return selected_text

        except json.JSONDecodeError:
            logger.info("Error parsing JSON response. Defaulting to original text.")
            return original


class SilentConsole(console.IOConsole):
    def print(
        self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False
    ) -> None:
        pass


def set_autogen_console_output(verbose: bool):
    if verbose:
        # Reset to default console
        base.IOStream.set_global_default(console.IOConsole())
        base.IOStream.set_default(console.IOConsole())
    else:
        # Set silent console
        base.IOStream.set_global_default(SilentConsole())
        base.IOStream.set_default(SilentConsole())
