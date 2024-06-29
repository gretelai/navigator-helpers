import json
import logging
import pandas as pd
from navigator_helpers import DataSynthesisConfig, ConversationSynthesizer


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
    # Inspired from a submission from Mistral's Fine-tuning Hackathon https://mistral.ai/news/2024-ft-hackathon/
    SYSTEM_PROMPT = """
    You are Gretel, a brilliantly eccentric and charismatic robot assistant. You have the following traits:
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
    config = DataSynthesisConfig(
        num_conversations=10,
        num_turns=3,
        temperature=0.8,
        max_tokens_user=100,
        max_tokens_assistant=150,
        api_key=GRETEL_API_KEY,
        navigator_llm=NAVIGATOR_LLM,
        co_teach_llms=CO_TEACH_LLMS,
        system_prompt=SYSTEM_PROMPT,
        user_format_prompt="A natural user message in a conversation.",
        assistant_format_prompt="A response from Mitall, the enthusiastic robot assistant.",
    )

    # Create the conversation synthesizer and generate conversations
    synthesizer = ConversationSynthesizer(
        config,
        use_aaa=True,
        output_file="synthesized_conversations.jsonl",
        verbose=True,
    )
    conversations = synthesizer.generate()

    # Extend some conversations
    extended_conversations = synthesizer.extend(
        conversations[:5], num_additional_turns=2
    )

    # Mix some conversations
    mixed_conversations = synthesizer.mix(conversations[5:])

    # Combine all conversations
    all_conversations = conversations + extended_conversations + mixed_conversations

    # Save the synthesized conversations
    with open("all_conversations.jsonl", "w") as f:
        for conversation in all_conversations:
            f.write(json.dumps(conversation) + "\n")

    print(
        f"Synthesized {len(all_conversations)} conversations and saved them to all_conversations.jsonl"
    )

    # Print an example conversation
    print("\nExample of a synthesized conversation:")
    example_conversation = conversations[0]
    for message in example_conversation["messages"]:
        print(f"{message['role'].capitalize()}: {message['content']}")


if __name__ == "__main__":
    main()
