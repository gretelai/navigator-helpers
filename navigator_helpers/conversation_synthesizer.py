import logging
import json
import re
from dataclasses import asdict, dataclass
from typing import List

from tqdm.auto import tqdm

from .aaa_utils import apply_aaa
from .data_synthesis import initialize_navigator, log_message, DataSynthesisConfig
from .evaluation_utils import evaluate_texts, rank_texts
from .prompt_templates import (
    CONV_ASSISTANT_MESSAGE_TEMPLATE,
    CONV_CO_TEACH_TEMPLATE,
    CONV_SELF_TEACHING_TEMPLATE,
    CONV_SUGGESTIONS_TEMPLATE,
    CONV_USER_MESSAGE_TEMPLATE,
)

logger = logging.getLogger(__name__)

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
    config: DataSynthesisConfig
    use_aaa: bool = True
    output_file: str = None
    verbose: bool = False

    def __post_init__(self):
        (
            self.navigator_llm,
            self.navigator_tabular,
            self.co_teach_llms,
        ) = initialize_navigator(self.config)

    def generate(self):
        if self.verbose:
            log_message("🚀 Starting conversation generation process")

        generated_conversations = []

        for i in tqdm(
            range(self.config.num_conversations),
            desc="Generating Conversations",
            disable=not self.verbose,
        ):
            if self.verbose:
                log_message(
                    f"\n📝 Generating conversation {i + 1}/{self.config.num_conversations}"
                )

            conversation = self._generate_conversation()
            conversation_dict = conversation.to_dict()

            if self.verbose:
                log_message(f"✅ Conversation {i + 1} generated successfully ⭐")
                self._print_conversation(conversation)

            if self.output_file:
                with open(self.output_file, "a") as f:
                    f.write(json.dumps(conversation_dict) + "\n")

            generated_conversations.append(conversation_dict)

        if self.verbose:
            log_message(
                f"🎉 Generated {self.config.num_conversations} conversations successfully 🎉"
            )

        return generated_conversations

    def _generate_conversation(self):
        conversation = Conversation(self.config.system_prompt)
        for turn in range(self.config.num_turns):
            user_message = self._generate_user_message(conversation)
            conversation.add_message("user", user_message)

            assistant_message = self._generate_assistant_message(conversation)
            conversation.add_message("assistant", assistant_message)

        return conversation

    def _generate_user_message(self, conversation):
        prompt = CONV_USER_MESSAGE_TEMPLATE.format(
            system_prompt=self.config.system_prompt,
            conversation_history=self._format_conversation(conversation),
            user_format_prompt=self.config.user_format_prompt,
        )
        message = self.navigator_llm.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_user,
        )

        if self.verbose:
            log_message(f"📝 Initial user message: {message}")

        if self.use_aaa:
            context = self._format_conversation(conversation)
            improved_message = apply_aaa(
                text=message,
                context=context,
                co_teach_llms=self.co_teach_llms,
                navigator_llm=self.navigator_llm,
                co_teach_template=CONV_CO_TEACH_TEMPLATE,
                suggestions_template=CONV_SUGGESTIONS_TEMPLATE,
                self_teaching_template=CONV_SELF_TEACHING_TEMPLATE,
                template_vars={"role": "user"},
                verbose=self.verbose,
            )
            messages = [message, improved_message]
            scores = evaluate_texts(
                texts=messages,
                column_name="text",
                additional_column="context",
                additional_value=context,
                format_prompt=self.config.user_format_prompt,
                navigator_tabular=self.navigator_tabular,
                verbose=self.verbose,
            )
            best_message = rank_texts(scores)
            message = best_message["text"]

            if self.verbose:
                log_message(f"✅ Improved user message: {message}")

        return message

    def _generate_assistant_message(self, conversation):
        prompt = CONV_ASSISTANT_MESSAGE_TEMPLATE.format(
            system_prompt=self.config.system_prompt,
            conversation_history=self._format_conversation(conversation),
            assistant_format_prompt=self.config.assistant_format_prompt,
        )
        message = self.navigator_llm.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_assistant,
        )

        if self.verbose:
            log_message(f"📝 Initial assistant message: {message}")

        if self.use_aaa:
            context = self._format_conversation(conversation)
            improved_message = apply_aaa(
                text=message,
                context=context,
                co_teach_llms=self.co_teach_llms,
                navigator_llm=self.navigator_llm,
                co_teach_template=CONV_CO_TEACH_TEMPLATE,
                suggestions_template=CONV_SUGGESTIONS_TEMPLATE,
                self_teaching_template=CONV_SELF_TEACHING_TEMPLATE,
                template_vars={"role": "assistant"},
                verbose=self.verbose,
            )
            messages = [message, improved_message]
            scores = evaluate_texts(
                texts=messages,
                column_name="text",
                additional_column="context",
                additional_value=context,
                format_prompt=self.config.assistant_format_prompt,
                navigator_tabular=self.navigator_tabular,
                verbose=self.verbose,
            )
            best_message = rank_texts(scores)
            message = best_message["text"]

            if self.verbose:
                log_message(f"✅ Improved assistant message: {message}")

        return message

    def _format_conversation(self, conversation):
        return "\n".join(
            [
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in conversation.messages
            ]
        )

    def _print_conversation(self, conversation):
        log_message("  Conversation:")
        for msg in conversation.messages:
            log_message(f"    {msg['role'].capitalize()}: {msg['content']}")
        log_message("")

def validate_generated_regex(dialog: dict) -> bool:
    if not isinstance(dialog, dict):
        return False

    dialog_str = json.dumps(dialog)
    pattern = r'^\s*\{"messages":\s*\[\s*\{"role":\s*"user",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\},\s*\{"role":\s*"assistant",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\}(?:,\s*\{"role":\s*"user",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\},\s*\{"role":\s*"assistant",\s*"content":\s*"[^"]*"(?:\\ "[^"]*")*\})*\s*\]\s*\}'

    return bool(re.match(pattern, dialog_str))