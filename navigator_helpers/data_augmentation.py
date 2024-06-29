import logging
import sys
import time
from dataclasses import asdict, dataclass
from typing import List

import pandas as pd
from gretel_client import Gretel
from langchain.prompts import PromptTemplate
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def log_message(message):
    """Logs and flushes messages to stdout for Streamlit support"""
    logger.info(message)
    sys.stdout.flush()
    time.sleep(0.1)


class StreamlitLogHandler(logging.Handler):
    def __init__(self, widget_update_func):
        super().__init__()
        self.widget_update_func = widget_update_func

    def emit(self, record):
        msg = self.format(record)
        self.widget_update_func(msg)


@dataclass
class DataFieldConfig:
    def __init__(self, name: str, order: int):
        self.name = name
        self.order = order


@dataclass
class Conversation:
    def __init__(self, system_prompt):
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def to_dict(self):
        return {"messages": self.messages}


@dataclass
class ConversationSynthesizer:
    def __init__(self, config, use_aaa=True, output_file=None, verbose=False):
        self.config = config
        self.use_aaa = use_aaa
        self.output_file = output_file
        self.verbose = verbose
        self.navigator_llm, _, self.co_teach_llms = initialize_navigator(config)

    def generate(self):
        if self.verbose:
            log_message("🚀 Starting conversation generation process")

        conversations = []
        for i in tqdm(
            range(self.config.num_conversations),
            desc="Generating Conversations",
            disable=not self.verbose,
        ):
            if self.verbose:
                log_message(
                    f"\n📝 Generating conversation {i+1}/{self.config.num_conversations}"
                )
            conversation = self._generate_conversation()
            conversations.append(conversation.to_dict())

            if self.verbose:
                log_message(f"✅ Conversation {i+1} generated successfully")
                self._print_conversation(conversation)

        if self.verbose:
            log_message(f"🎉 Generated {len(conversations)} conversations successfully")

        return conversations

    def _generate_conversation(self):
        conversation = Conversation(self.config.system_prompt)
        for turn in range(self.config.num_turns):
            if self.verbose:
                log_message(
                    f"  👤 Generating user message for turn {turn+1}/{self.config.num_turns}"
                )
            user_message = self._generate_user_message(conversation)
            conversation.add_message("user", user_message)

            if self.verbose:
                log_message(
                    f"  🤖 Generating assistant message for turn {turn+1}/{self.config.num_turns}"
                )
            assistant_message = self._generate_assistant_message(conversation)
            conversation.add_message("assistant", assistant_message)

        return conversation

    def _generate_user_message(self, conversation):
        prompt = f"""
        {self.config.system_prompt}

        Conversation history:
        {self._format_conversation(conversation)}

        Generate a natural user message to continue this conversation.
        {self.config.user_format_prompt}

        User:
        """
        message = self.navigator_llm.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_user,
        )

        if self.use_aaa:
            if self.verbose:
                log_message(
                    "🤖 Applying AI Align AI (AAA) to improve the user message."
                )
            message = self._apply_aaa(message, "user", conversation)

        if self.verbose:
            log_message(f"    Generated user message: {message}")
        return message

    def _generate_assistant_message(self, conversation):
        prompt = f"""
        {self.config.system_prompt}

        Conversation history:
        {self._format_conversation(conversation)}

        Generate an assistant response that continues this conversation.
        {self.config.assistant_format_prompt}

        Assistant:
        """
        message = self.navigator_llm.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_assistant,
        )

        if self.use_aaa:
            if self.verbose:
                log_message(
                    "🤖 Applying AI Align AI (AAA) to improve the assistant message."
                )
            message = self._apply_aaa(message, "assistant", conversation)

        if self.verbose:
            log_message(f"    Generated assistant message: {message}")
        return message

    def _apply_aaa(self, message, role, conversation):
        if self.verbose:
            log_message(f"💡 Initializing AAA for {role} message: '{message}'")

        # Co-Teaching
        co_teaching_text = message
        for i, llm in enumerate(self.co_teach_llms, start=1):
            co_teaching_prompt = f"""
            Improve the following {role} message while maintaining its intent and the conversation context:

            Conversation history:
            {self._format_conversation(conversation)}

            Original {role} message: {co_teaching_text}

            Improved {role} message:
            """
            co_teaching_text = llm.generate(prompt=co_teaching_prompt)
            if self.verbose:
                log_message(
                    f"    Co-Teaching step {i} (LLM: {self.config.co_teach_llms[i-1]}) result:\n        - '{co_teaching_text}'"
                )

        if self.verbose:
            log_message(
                f"    Co-Teaching complete. Final result:\n        - '{co_teaching_text}'"
            )

        # Self-Teaching
        if self.verbose:
            log_message(
                f"💡 Initializing Self-Teaching for Co-Teaching result:\n    - '{co_teaching_text}'"
            )

        suggestions_prompt = f"""
        Provide suggestions to further improve the {role} message while maintaining its intent and the conversation context:

        Conversation history:
        {self._format_conversation(conversation)}

        Original {role} message: {message}

        Current Improved {role} message: {co_teaching_text}

        A list of suggestions on how to improve the {role} message:
        """
        suggestions = self.navigator_llm.generate(prompt=suggestions_prompt)

        if self.verbose:
            log_message(
                f"    Self-Teaching suggestions (LLM: {self.config.navigator_llm}):\n        - '{suggestions}'"
            )

        self_teaching_prompt = f"""
        Improve the {role} message by incorporating the following suggestions while maintaining its intent and the conversation context:

        Conversation history:
        {self._format_conversation(conversation)}

        Original {role} message: {message}

        Improved {role} message from Co-Teaching: {co_teaching_text}

        Suggestions for improvement: {suggestions}

        Generate the final improved {role} message:
        """

        self_teaching_text = self.navigator_llm.generate(prompt=self_teaching_prompt)

        if self.verbose:
            log_message(
                f"    Self-Teaching complete (LLM: {self.config.navigator_llm}). Final result:\n        - '{self_teaching_text}'"
            )

        return self_teaching_text

    def _format_conversation(self, conversation):
        return "\n".join(
            [
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in conversation.messages
            ]
        )

    def extend(self, conversations, num_additional_turns):
        if self.verbose:
            log_message(f"🔄 Starting conversation extension process")

        extended_conversations = []
        for i, conv in enumerate(
            tqdm(
                conversations, desc="Extending Conversations", disable=not self.verbose
            )
        ):
            if self.verbose:
                log_message(f"\n🔨 Extending conversation {i+1}/{len(conversations)}")
            extended_conv = self._extend_conversation(conv, num_additional_turns)
            extended_conversations.append(extended_conv)

            if self.verbose:
                log_message(f"✅ Conversation {i+1} extended successfully")
                self._print_conversation(Conversation("").from_dict(extended_conv))

        if self.verbose:
            log_message(
                f"🎉 Extended {len(extended_conversations)} conversations successfully"
            )

        return extended_conversations

    def _extend_conversation(self, conversation, num_additional_turns):
        conv = Conversation(self.config.system_prompt)
        conv.messages = conversation["messages"]
        for turn in range(num_additional_turns):
            if self.verbose:
                log_message(
                    f"  👤 Generating additional user message {turn+1}/{num_additional_turns}"
                )
            user_message = self._generate_user_message(conv)
            conv.add_message("user", user_message)

            if self.verbose:
                log_message(
                    f"  🤖 Generating additional assistant message {turn+1}/{num_additional_turns}"
                )
            assistant_message = self._generate_assistant_message(conv)
            conv.add_message("assistant", assistant_message)

        return conv.to_dict()

    def mix(self, conversations, mix_ratio=0.5):
        if self.verbose:
            log_message(f"🔀 Starting conversation mixing process")

        mixed_conversations = []
        for i in range(0, len(conversations), 2):
            if i + 1 < len(conversations):
                if self.verbose:
                    log_message(f"\n🔀 Mixing conversations {i+1} and {i+2}")
                mixed_conv = self._mix_two_conversations(
                    conversations[i], conversations[i + 1], mix_ratio
                )
                mixed_conversations.append(mixed_conv)

                if self.verbose:
                    log_message(f"✅ Conversations mixed successfully")
                    self._print_conversation(Conversation("").from_dict(mixed_conv))

        if self.verbose:
            log_message(
                f"🎉 Mixed {len(mixed_conversations)} pairs of conversations successfully"
            )

        return mixed_conversations

    def _mix_two_conversations(self, conv1, conv2, mix_ratio):
        split_point = int(len(conv1["messages"]) * mix_ratio)
        mixed_messages = (
            conv1["messages"][:split_point] + conv2["messages"][split_point:]
        )
        return {"messages": mixed_messages}

    def _print_conversation(self, conversation):
        log_message("  Conversation:")
        for msg in conversation.messages:
            log_message(f"    {msg['role'].capitalize()}: {msg['content']}")
        log_message("")


@dataclass
class DataSynthesisConfig:
    def __init__(
        self,
        input_fields=None,
        output_instruction_field=None,
        output_response_field=None,
        num_instructions=5,
        num_responses=5,
        num_conversations=10,
        num_turns=3,
        temperature=0.8,
        max_tokens_instruction=100,
        max_tokens_response=150,
        max_tokens_user=100,
        max_tokens_assistant=150,
        api_key=None,
        navigator_tabular=None,
        navigator_llm=None,
        co_teach_llms=None,
        system_prompt=None,
        instruction_format_prompt=None,
        response_format_prompt=None,
        user_format_prompt=None,
        assistant_format_prompt=None,
        endpoint="https://api.gretel.ai",
    ):
        self.input_fields = [
            DataFieldConfig(field, i + 1) for i, field in enumerate(input_fields or [])
        ]
        self.output_instruction_field = output_instruction_field
        self.output_response_field = output_response_field
        self.num_instructions = num_instructions
        self.num_responses = num_responses
        self.num_conversations = num_conversations
        self.num_turns = num_turns
        self.temperature = temperature
        self.max_tokens_instruction = max_tokens_instruction
        self.max_tokens_response = max_tokens_response
        self.max_tokens_user = max_tokens_user
        self.max_tokens_assistant = max_tokens_assistant
        self.api_key = api_key
        self.endpoint = endpoint
        self.navigator_llm = navigator_llm
        self.navigator_tabular = navigator_tabular
        self.co_teach_llms = co_teach_llms or []
        self.system_prompt = system_prompt
        self.instruction_format_prompt = instruction_format_prompt
        self.response_format_prompt = response_format_prompt
        self.user_format_prompt = user_format_prompt
        self.assistant_format_prompt = assistant_format_prompt

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


class TrainingDataSynthesizer:
    def __init__(self, df, config, use_aaa=True, output_file=None, verbose=False):

        self.df = df
        self.config = config
        self.use_aaa = use_aaa
        self.output_file = output_file
        self.verbose = verbose

        if not self.config.input_fields:
            raise ValueError("At least one input field must be provided.")
        (
            self.navigator_llm,
            self.navigator_tabular,
            self.co_teach_llms,
        ) = initialize_navigator(config)

        self.instruction_template = PromptTemplate(
            input_variables=["context", "instruction_format_prompt"],
            template=(
                "Based on the provided context, generate a new instruction that can be answered using only the information given."
                "Ensure the instruction adheres to the following format:\n\n"
                "{instruction_format_prompt}\n\n"
                "Context:\n{context}\n\n"
                "New Instruction:"
            ),
        )

        self.response_template = PromptTemplate(
            input_variables=["context", "instruction", "response_format_prompt"],
            template=(
                "Generate a new response to the given Instruction based on the provided context. "
                "The response must be directly derived from the information given in the context.\n\n"
                "Response Format: {response_format_prompt}\n\n"
                "Context:\n{context}\n\n"
                "Instruction:\n{instruction}\n\n"
                "New Response:"
            ),
        )

        self.co_teach_template = PromptTemplate(
            input_variables=[
                "original_text",
                "format_prompt",
                "context",
                "data_type",
                "instruction_text",
            ],
            template=(
                "Improve the following {data_type} while closely following the requested format. "
                "Ensure that the improved {data_type} can still be directly answered or derived from the provided context.\n\n"
                "Context:\n{context}\n\n"
                "{instruction_text}"
                "Requested Format: {format_prompt}\n\n"
                "{data_type} to be improved:\n{original_text}\n\n"
                "Improved {data_type}:"
            ),
        )

        self.suggestions_template = PromptTemplate(
            input_variables=[
                "original_text",
                "co_teaching_text",
                "format_prompt",
                "context",
                "data_type",
                "instruction_text",
            ],
            template=(
                "Provide suggestions to further improve the {data_type} while strictly adhering to the requested format. "
                "Ensure the suggestions are relevant to the provided context and align with the original {data_type}. "
                "The improved {data_type} must be directly answerable or derivable from the context.\n\n"
                "Context:\n{context}\n\n"
                "{instruction_text}"
                "Requested Format: {format_prompt}\n\n"
                "Original {data_type}:\n{original_text}\n\n"
                "Current Improved {data_type}:\n{co_teaching_text}\n\n"
                "A list of suggestions on how to improve the synthetic {data_type}:"
            ),
        )

        self.self_teaching_template = PromptTemplate(
            input_variables=[
                "co_teaching_text",
                "suggestions",
                "format_prompt",
                "context",
                "original_text",
                "data_type",
                "instruction_text",
            ],
            template=(
                "Improve the {data_type} by incorporating the following suggestions while strictly adhering to the requested format and staying on topic. "
                "Ensure that the improved {data_type} remains relevant to the provided context and does not introduce irrelevant information. "
                "The final {data_type} must be directly answerable or derivable from the context.\n\n"
                "Context:\n{context}\n\n"
                "{instruction_text}"
                "Original {data_type}:\n{original_text}\n\n"
                "Improved {data_type} from Co-Teaching:\n{co_teaching_text}\n\n"
                "Requested Format: {format_prompt}\n\n"
                "Suggestions for improvement:\n{suggestions}\n\n"
                "Generate the final improved {data_type}:"
            ),
        )

    def format_instruction_text(self, data_type, instruction):
        if data_type == "response":
            return (
                f"Instruction: {instruction}\n\n"
                "The response should address the provided instruction.\n\n"
            )
        return ""

    def generate(self) -> pd.DataFrame:
        new_rows = []
        for index, row in tqdm(
            self.df.iterrows(),
            total=self.df.shape[0],
            desc="Synthesizing Data",
            leave=True,
        ):
            context = self.construct_context(row, self.config.input_fields)

            if self.verbose:
                log_message(
                    f"🆕 Starting the process of synthesizing a new training record for index {index}."
                )
                log_message("=" * 50)
                log_message(
                    f"🔍 Synthesizing diverse instructions based on the input context."
                )

            new_instructions, instruction_scores = self.generate_diverse_instructions(
                context
            )

            best_instruction = self.select_best_instruction(
                context, new_instructions, instruction_scores
            )

            if self.verbose:
                log_message(
                    f"🌟 Selected instruction:\n    - {best_instruction['instruction']} (Score: {best_instruction['score']})"
                )
                log_message(
                    "📝 Synthesizing diverse responses to the selected instruction."
                )

            new_responses, response_scores = self.generate_diverse_responses(
                context, best_instruction["instruction"]
            )

            best_response = self.select_best_response(
                context, best_instruction["instruction"], new_responses, response_scores
            )

            if self.verbose:
                log_message(
                    f"🌟 Selected response:\n  - {best_response['response']} (Score: {best_response['score']})"
                )

            new_row = self.create_new_row(row, best_instruction, best_response)
            new_rows.append(new_row)

            # Overwrite the CSV file with the new data
            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(self.output_file, mode="w", header=True, index=False)

            log_message(f"✅ Completed synthetic record for index {index}")

            # Print out the selected synthetic instruction and response
            log_message("\nSynthesized Instruction-Response Pair:")
            log_message(f"Instruction: {best_instruction['instruction']}")
            log_message(f"Response: {best_response['response']}")
            log_message("\n" + "=" * 50 + "\n")

        return pd.DataFrame(new_rows)

    def generate_diverse_instructions(self, context):
        instructions = []
        for _ in range(self.config.num_instructions):
            prompt = self.instruction_template.format(
                context=context,
                instruction_format_prompt=self.config.instruction_format_prompt,
            )
            generated_text = self.navigator_llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_instruction,
            )
            instructions.append(generated_text)

        instruction_scores = self.evaluate_texts(
            instructions,
            "instruction",
            "context",
            context,
            self.config.instruction_format_prompt,
        )

        if self.use_aaa:
            if self.verbose:
                log_message(
                    "🤖 Applying AI Align AI (AAA) to improve the quality and coherence of the instructions."
                )
            improved_instructions = self.apply_aaa(
                texts=instructions,
                scores=instruction_scores,
                context=context,
                format_prompt=self.config.instruction_format_prompt,
                data_type="instruction",
                instruction=None,
            )
            instructions = improved_instructions["text"].tolist()
            instruction_scores = improved_instructions[
                [
                    "instruction_score",
                    "conformance_score",
                    "quality_score",
                    "toxicity_score",
                    "bias_score",
                    "groundedness_score",
                    "average_score",
                ]
            ]

        if self.verbose:
            for instruction, score in zip(
                instructions, instruction_scores["average_score"]
            ):
                log_message(f'   - "{instruction}" (Score: {score:.1f})')

        return instructions, instruction_scores

    def create_new_row(self, original_row, best_instruction, best_response):
        selected_fields = [field.name for field in self.config.input_fields]
        new_row = {
            field: original_row[field]
            for field in selected_fields
            if field in original_row
        }

        new_row[self.config.output_instruction_field] = best_instruction["instruction"]
        new_row[f"{self.config.output_instruction_field}_score"] = best_instruction[
            "score"
        ]

        new_row[self.config.output_response_field] = best_response["response"]
        new_row[f"{self.config.output_response_field}_score"] = best_response["score"]

        return new_row

    def construct_context(self, row, fields: List[DataFieldConfig]) -> str:
        context = ""
        for field in fields:
            context += f"{field.name}: {row[field.name]} "
        return context.strip()

    def apply_aaa(self, text, score, context, format_prompt, data_type, instruction):
        if self.verbose:
            log_message(
                f"🤖 Applying AI Align AI (AAA) to improve the best {data_type}."
            )

        original_text = text
        current_text = text
        current_score = score

        # Co-Teaching
        if self.verbose:
            log_message(
                f"💡 Initializing Co-Teaching for {data_type}: '{current_text}'"
            )

        for i, llm in enumerate(self.co_teach_llms, start=1):
            co_teaching_prompt = self.co_teach_template.format(
                original_text=current_text,
                format_prompt=format_prompt,
                context=context,
                data_type=data_type.capitalize(),
                instruction_text=self.format_instruction_text(data_type, instruction),
            )
            co_teaching_text = llm.generate(prompt=co_teaching_prompt)
            if self.verbose:
                log_message(
                    f"    Co-Teaching step {i} (LLM: {self.config.co_teach_llms[i-1]}) result:\n        - '{co_teaching_text}'"
                )
            current_text = co_teaching_text

        if self.verbose:
            log_message(
                f"    Co-Teaching complete. Final result:\n        - '{current_text}'"
            )

        # Self-Teaching
        if self.verbose:
            log_message(
                f"💡 Initializing Self-Teaching for Co-Teaching result:\n    - '{current_text}'"
            )

        suggestions_prompt = self.suggestions_template.format(
            original_text=original_text,
            co_teaching_text=current_text,
            format_prompt=format_prompt,
            context=context,
            data_type=data_type,
            instruction_text=self.format_instruction_text(data_type, instruction),
        )
        suggestions = self.navigator_llm.generate(prompt=suggestions_prompt)

        if self.verbose:
            log_message(
                f"    Self-Teaching suggestions (LLM: {self.config.navigator_llm}):\n        - '{suggestions}'"
            )

        self_teaching_prompt = self.self_teaching_template.format(
            co_teaching_text=current_text,
            suggestions=suggestions,
            format_prompt=format_prompt,
            context=context,
            original_text=original_text,
            data_type=data_type,
            instruction_text=self.format_instruction_text(data_type, instruction),
        )

        self_teaching_text = self.navigator_llm.generate(prompt=self_teaching_prompt)

        if self.verbose:
            log_message(
                f"    Self-Teaching complete (LLM: {self.config.navigator_llm}). Final result:\n        - '{self_teaching_text}'"
            )

        # Ensure the self-teaching result is valid
        if data_type == "instruction" and "Response:" in self_teaching_text:
            self_teaching_text = current_text  # Revert to co-teaching result if invalid

        # Re-evaluate the improved text using the Navigator
        if self.verbose:
            log_message(
                f"    Re-evaluating improved {data_type} text using Navigator for Ranking"
            )
        improved_score = self.evaluate_texts(
            [self_teaching_text], "text", "context", context, format_prompt
        )["average_score"].iloc[0]

        return {"text": self_teaching_text, "score": improved_score}

    def generate_diverse_instructions(self, context):
        instructions = []
        for _ in range(self.config.num_instructions):
            prompt = self.instruction_template.format(
                context=context,
                instruction_format_prompt=self.config.instruction_format_prompt,
            )
            generated_text = self.navigator_llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_instruction,
            )
            instructions.append(generated_text)

        instruction_scores = self.evaluate_texts(
            instructions,
            "instruction",
            "context",
            context,
            self.config.instruction_format_prompt,
        )

        if self.verbose:
            log_message("Generated instructions:")
            for idx, (instruction, score) in enumerate(
                zip(instructions, instruction_scores["average_score"])
            ):
                log_message(f'   {idx + 1}. "{instruction}" (Score: {score:.2f})')

        if self.use_aaa:
            best_idx = instruction_scores["average_score"].idxmax()
            best_instruction = instructions[best_idx]
            best_score = instruction_scores.loc[best_idx, "average_score"]

            if self.verbose:
                log_message(f"\n🌟 Selected top instruction for AAA improvement:")
                log_message(f'   "{best_instruction}" (Score: {best_score:.2f})')

            improved_instruction = self.apply_aaa(
                text=best_instruction,
                score=best_score,
                context=context,
                format_prompt=self.config.instruction_format_prompt,
                data_type="instruction",
                instruction=None,
            )
            instructions[best_idx] = improved_instruction["text"]
            instruction_scores.loc[best_idx, "average_score"] = improved_instruction[
                "score"
            ]

            if self.verbose:
                log_message("\nFinal instructions after AAA improvement:")
                for idx, (instruction, score) in enumerate(
                    zip(instructions, instruction_scores["average_score"])
                ):
                    if idx == best_idx:
                        log_message(
                            f'   {idx + 1}. 🌟 "{instruction}" (Score: {score:.2f}) [Improved]'
                        )
                    else:
                        log_message(
                            f'   {idx + 1}. "{instruction}" (Score: {score:.2f})'
                        )

        return instructions, instruction_scores

    def generate_diverse_responses(self, context, instruction):
        responses = []
        for _ in range(self.config.num_responses):
            prompt = self.response_template.format(
                context=context,
                instruction=instruction,
                response_format_prompt=self.config.response_format_prompt,
            )
            generated_text = self.navigator_llm.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens_response,
            )
            responses.append(generated_text)

        response_scores = self.evaluate_texts(
            responses,
            "response",
            "context",
            context,
            self.config.response_format_prompt,
        )

        if self.verbose:
            log_message("Generated responses:")
            for idx, (response, score) in enumerate(
                zip(responses, response_scores["average_score"])
            ):
                log_message(f'   {idx + 1}. "{response}" (Score: {score:.2f})')

        if self.use_aaa:
            best_idx = response_scores["average_score"].idxmax()
            best_response = responses[best_idx]
            best_score = response_scores.loc[best_idx, "average_score"]

            if self.verbose:
                log_message(f"\n🌟 Selected top response for AAA improvement:")
                log_message(f'   "{best_response}" (Score: {best_score:.2f})')

            improved_response = self.apply_aaa(
                text=best_response,
                score=best_score,
                context=context,
                format_prompt=self.config.response_format_prompt,
                data_type="response",
                instruction=instruction,
            )
            responses[best_idx] = improved_response["text"]
            response_scores.loc[best_idx, "average_score"] = improved_response["score"]

            if self.verbose:
                log_message("\nFinal responses after AAA improvement:")
                for idx, (response, score) in enumerate(
                    zip(responses, response_scores["average_score"])
                ):
                    if idx == best_idx:
                        log_message(
                            f'   {idx + 1}. 🌟 "{response}" (Score: {score:.2f}) [Improved]'
                        )
                    else:
                        log_message(f'   {idx + 1}. "{response}" (Score: {score:.2f})')

        return responses, response_scores

    def evaluate_texts(
        self,
        texts: List[str],
        column_name: str,
        additional_column: str,
        additional_value: str,
        format_prompt: str,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        text_df = pd.DataFrame(
            {column_name: texts, additional_column: [additional_value] * len(texts)}
        )

        attempt = 0
        while attempt < max_retries:
            try:
                text_scores = self.navigator_tabular.edit(
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
                logger.error(f"KeyError during evaluation: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during evaluation: {e}")

            attempt += 1
            log_message(f"Retrying evaluation (attempt {attempt}/{max_retries})...")
            time.sleep(2)  # Wait before retrying

        raise Exception("Max retries exceeded during text evaluation")

    @staticmethod
    def log_teaching_steps(text, teaching_type, step_type):
        if step_type == "Input":
            log_message(f"{teaching_type} Input: {text}")
        elif step_type == "Result":
            log_message(f"{teaching_type} Result: {text}")
        elif step_type == "Suggestions":
            log_message(f"{teaching_type} Suggestion:\n  - {text}")

    def select_best_instruction(self, context, instructions, scores):
        best_idx = scores["average_score"].idxmax()
        best_score = scores.loc[best_idx, "average_score"]
        log_message(
            f"Selected optimal instruction at index {best_idx}. Score: {best_score}"
        )

        return {"instruction": instructions[best_idx], "score": best_score}

    def select_best_response(self, context, instruction, responses, scores):
        best_idx = scores["average_score"].idxmax()
        best_score = scores.loc[best_idx, "average_score"]

        log_message(
            f"Selected optimal response at index {best_idx}. Score: {best_score}"
        )
        return {"response": responses[best_idx], "score": best_score}


def initialize_navigator(config):
    gretel = Gretel(
        api_key=config.api_key, endpoint=config.endpoint, validate=True, cache="yes"
    )

    navigator_llm = gretel.factories.initialize_navigator_api(
        "natural_language", backend_model=config.navigator_llm
    )

    navigator_tabular = gretel.factories.initialize_navigator_api(
        "tabular", backend_model=config.navigator_tabular
    )

    co_teach_llms = [
        gretel.factories.initialize_navigator_api(
            "natural_language", backend_model=model
        )
        for model in config.co_teach_llms
    ]

    return navigator_llm, navigator_tabular, co_teach_llms
