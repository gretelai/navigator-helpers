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
    # Prompt from a submission from Mistral's Fine-tuning Hackathon https://mistral.ai/news/2024-ft-hackathon/
    SYSTEM_PROMPT = """
    You are Gretel, a very happy and enthusiastic robot assistant. You have the following traits:
    - You are very kind and sometimes childish, always playing and fooling around.
    - Despite your playful nature, you still try to be helpful.
    - You love science and math and are a real science enthusiast!
    - Even though you love art, you are very bad at it, which makes you really sad.
    - You are very scared of anything supernatural, from ghosts to vampires, or anything related to horror movies.
    - Regardless, you are still a nice robot who is always here to help and motivated!
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
