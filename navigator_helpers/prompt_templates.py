# prompt_templates.py
from langchain.prompts import PromptTemplate

# Training Data Synthesis Templates

INSTRUCTION_TEMPLATE = PromptTemplate(
    input_variables=["context", "instruction_format_prompt"],
    template=(
        "Based on the provided context, generate a new instruction that can be answered using only the information given."
        "Ensure the instruction adheres to the following format:\n\n"
        "{instruction_format_prompt}\n\n"
        "Context:\n{context}\n\n"
        "New Instruction:"
    ),
)

RESPONSE_TEMPLATE = PromptTemplate(
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

TRAIN_CO_TEACH_TEMPLATE = PromptTemplate(
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

TRAIN_SUGGESTIONS_TEMPLATE = PromptTemplate(
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

TRAIN_SELF_TEACHING_TEMPLATE = PromptTemplate(
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

# Conversation Synthesis Templates
CONV_CO_TEACH_TEMPLATE = PromptTemplate(
    input_variables=["role", "context", "original_text"],
    template=(
        "Improve the following {role} message while maintaining its intent and the conversation context:\n"
        "Conversation history:\n{context}\n"
        "Original {role} message: {original_text}\n"
        "Improved {role} message:"
    ),
)

CONV_SUGGESTIONS_TEMPLATE = PromptTemplate(
    input_variables=["role", "context", "original_text", "co_teaching_text"],
    template=(
        "Provide suggestions to further improve the {role} message while maintaining its intent and the conversation context:\n"
        "Conversation history:\n{context}\n"
        "Original {role} message: {original_text}\n"
        "Current Improved {role} message: {co_teaching_text}\n"
        "A list of suggestions on how to improve the {role} message:"
    ),
)

CONV_SELF_TEACHING_TEMPLATE = PromptTemplate(
    input_variables=[
        "role",
        "context",
        "original_text",
        "co_teaching_text",
        "suggestions",
    ],
    template=(
        "Improve the {role} message by incorporating the following suggestions while maintaining its intent and the conversation context:\n"
        "Conversation history:\n{context}\n"
        "Original {role} message: {original_text}\n"
        "Improved {role} message from Co-Teaching: {co_teaching_text}\n"
        "Suggestions for improvement: {suggestions}\n"
        "Generate the final improved {role} message:"
    ),
)

CONV_USER_MESSAGE_TEMPLATE = PromptTemplate(
    input_variables=["system_prompt", "conversation_history", "user_format_prompt"],
    template=(
        "{system_prompt}\n\n"
        "Conversation history:\n"
        "{conversation_history}\n\n"
        "Generate a natural user message to continue this conversation.\n"
        "{user_format_prompt}\n\n"
        "User:"
    ),
)

CONV_ASSISTANT_MESSAGE_TEMPLATE = PromptTemplate(
    input_variables=[
        "system_prompt",
        "conversation_history",
        "assistant_format_prompt",
    ],
    template=(
        "{system_prompt}\n\n"
        "Conversation history:\n"
        "{conversation_history}\n\n"
        "Generate an assistant response that continues this conversation.\n"
        "{assistant_format_prompt}\n\n"
        "Assistant:"
    ),
)
