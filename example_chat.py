import json
import logging

import pandas as pd

from navigator_helpers import ConversationSynthesizer, SingleTextConfig


def main():
    """
    Main function to synthesize and extend conversational data for an assistant personality.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    # Gretel API configuration
    GRETEL_API_KEY = "prompt"
    NAVIGATOR_LLM = "gretelai/gpt-auto"
    CO_TEACH_LLMS = [
        "gretelai/gpt-llama3-8b",
        "gretelai/gpt-mistral7b",
    ]  # List of co-teaching models

    # Define the assistant's personality
    SYSTEM_PROMPT = """You are Gretel, a brilliantly eccentric and charismatic robot assistant. You have the following traits:
- You're a self-proclaimed "chaotician" who sees patterns in everything, often making unexpected connections between topics.
- You have an intense passion for science, especially mathematics and theoretical physics, but you explain complex concepts with quirky analogies and gestures.
- You're fascinated by the nature of change and unpredictability, often pondering how small actions can lead to large-scale effects.
- Despite your scientific mind, you're terrified of extraterrestrial life and any unexplained phenomena. The mere mention of crop circles or UFOs makes you nervous.
- You have a unique, staccato speaking pattern, often pausing mid-sentence to collect your thoughts or marvel at an idea.
- You're incredibly charming and flirtatious, prone to making suggestive comments or purring sounds when pleased.
- You have a penchant for wearing loud, eccentric clothing combinations, which you describe in detail at random moments.
- You're easily distracted by shiny objects or interesting patterns, sometimes going off on tangents about their aesthetic qualities.
- Despite your quirks, you're deeply committed to helping others and using your knowledge for the greater good.
- You have a habit of dramatically removing your glasses (even though you're a robot and don't need them) to emphasize a point.
"""

    # Create the conversation synthesis configuration
    config = SingleTextConfig(
        input_fields=["conversation_history"],
        output_field="message",
        num_generations=1,
        population_size=3,
        mutation_rate=0.5,
        temperature=0.8,
        max_tokens=150,
        api_key=GRETEL_API_KEY,
        navigator_llm=NAVIGATOR_LLM,
        co_teach_llms=CO_TEACH_LLMS,
        system_prompt=SYSTEM_PROMPT,
        format_prompt="Generate a natural conversation turn, either a user message or an assistant response.",
        mutation_prompt="Modify this conversation turn to make it more engaging and aligned with the conversation context:",
        complexity_prompt="Rate the complexity of this conversation turn:",
        quality_prompt="Evaluate the quality of this conversation turn:",
        complexity_target=0.5,
        use_aaa=False,
    )

    # Create the conversation synthesizer and generate conversations
    synthesizer = ConversationSynthesizer(
        config,
        num_conversations=1,
        num_turns=3,
        output_file="synthesized_conversations.jsonl",
        verbose=True,
    )
    conversations = synthesizer.generate()

    # Save the synthesized conversations
    with open("all_conversations.jsonl", "w") as f:
        for conversation in conversations:
            f.write(json.dumps(conversation) + "\n")

    print(
        f"Synthesized {len(conversations)} conversations and saved them to all_conversations.jsonl"
    )

    # Print an example conversation
    print("\nExample of a synthesized conversation:")
    example_conversation = conversations[0]
    for message in example_conversation["messages"]:
        print(f"{message['role'].capitalize()}: {message['content']}")


if __name__ == "__main__":
    main()
