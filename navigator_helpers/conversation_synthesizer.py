import json
import logging
from dataclasses import dataclass
from typing import List

from tqdm.auto import tqdm

from .data_synthesis import SingleTextConfig, initialize_navigator
from .generation_types import GenerationType
from .text_generation import EvolutionaryTextGenerator, log_message

logger = logging.getLogger(__name__)


@dataclass
class Conversation:
    def __init__(self, system_prompt):
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def to_dict(self):
        return {"messages": self.messages}


class ConversationSynthesizer:
    def __init__(
        self,
        config: SingleTextConfig,
        num_conversations: int,
        num_turns: int,
        output_file: str = None,
        verbose: bool = False,
    ):
        self.config = config
        self.num_conversations = num_conversations
        self.num_turns = num_turns
        self.output_file = output_file
        self.verbose = verbose

        (
            self.navigator_llm,
            self.navigator_tabular,
            self.co_teach_llms,
        ) = initialize_navigator(self.config)
        self.text_generator = EvolutionaryTextGenerator(
            llm=self.navigator_llm,
            co_teach_llms=self.co_teach_llms,
            config=self.config,
            verbose=self.verbose,
        )

    def generate(self):
        if self.verbose:
            log_message("🚀 Starting conversation generation process")

        generated_conversations = []

        for i in tqdm(
            range(self.num_conversations),
            desc="Generating Conversations",
            disable=not self.verbose,
        ):
            if self.verbose:
                log_message(
                    f"\n📝 Generating conversation {i + 1}/{self.num_conversations}"
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
                f"🎉 Generated {self.num_conversations} conversations successfully 🎉"
            )

        return generated_conversations

    def _generate_conversation(self):
        conversation = Conversation(self.config.system_prompt)
        for turn in range(self.num_turns):
            user_message = self._generate_message(conversation, "user")
            conversation.add_message("user", user_message)

            assistant_message = self._generate_message(conversation, "assistant")
            conversation.add_message("assistant", assistant_message)

        return conversation

    def _generate_message(self, conversation, role):
        context = self._format_conversation(conversation)
        context += f"\n\nGenerate a {role} message."

        message = self.text_generator.generate(
            context=context, generation_type=GenerationType.TEXT
        )

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
