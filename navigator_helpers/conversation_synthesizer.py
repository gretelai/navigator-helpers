import logging
from dataclasses import asdict, dataclass
from typing import List

from tqdm.auto import tqdm

from .aaa_utils import apply_aaa
from .data_synthesis import initialize_navigator, log_message
from .evaluation_utils import evaluate_texts, rank_texts
from .prompt_templates import (CONV_ASSISTANT_MESSAGE_TEMPLATE,
                               CONV_CO_TEACH_TEMPLATE,
                               CONV_SELF_TEACHING_TEMPLATE,
                               CONV_SUGGESTIONS_TEMPLATE,
                               CONV_USER_MESSAGE_TEMPLATE)

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
    def __init__(self, config, use_aaa=True, output_file=None, verbose=False):
        self.config = config
        self.use_aaa = use_aaa
        self.output_file = output_file
        self.verbose = verbose
        (
            self.navigator_llm,
            self.navigator_tabular,
            self.co_teach_llms,
        ) = initialize_navigator(config)

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
            log_message(f"    Generated initial user message: {message}")

        if self.use_aaa:
            if self.verbose:
                log_message("🤖 Applying AI Align AI (AAA) to improve the user message.")

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

            # Evaluate both the original and improved messages
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

            # Select the best message
            best_message = rank_texts(scores)
            message = best_message["text"]

            if self.verbose:
                log_message(f"    Final user message: {message}")
                log_message(f"    Score: {best_message['score']:.2f}")

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
            log_message(f"    Generated initial assistant message: {message}")

        if self.use_aaa:
            if self.verbose:
                log_message(
                    "🤖 Applying AI Align AI (AAA) to improve the assistant message."
                )

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

            # Evaluate both the original and improved messages
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

            # Select the best message
            best_message = rank_texts(scores)
            message = best_message["text"]

            if self.verbose:
                log_message(f"    Final assistant message: {message}")
                log_message(f"    Score: {best_message['score']:.2f}")

        return message

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
