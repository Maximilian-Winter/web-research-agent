# Defaults
import datetime

import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Locals
from llama_cpp_agent.providers import LlamaCppPythonProvider, LlamaCppServerProvider
from content import css, PLACEHOLDER
from utils import CitingSources
# Agents
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent.chat_history import BasicChatHistory
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_output_settings import (
    LlmStructuredOutputSettings,
    LlmStructuredOutputType,
)
# Tools
from llama_cpp_agent.tools import WebSearchTool
from llama_cpp_agent.prompt_templates import web_search_system_prompt, research_system_prompt





def get_context_by_model(model_name):
    model_context_limits = {
        "Mistral-7B-Instruct-v0.3-Q6_K.gguf": 32768,
        "Meta-Llama-3-8B-Instruct-Q6_K.gguf": 8192
    }
    return model_context_limits.get(model_name, None)


def get_messages_formatter_type(model_name):
    from llama_cpp_agent import MessagesFormatterType
    if "Meta" in model_name or "aya" in model_name:
        return MessagesFormatterType.LLAMA_3
    elif "Mistral" in model_name:
        return MessagesFormatterType.MISTRAL
    elif "Einstein-v6-7B" in model_name or "dolphin" in model_name:
        return MessagesFormatterType.CHATML
    elif "Phi" in model_name:
        return MessagesFormatterType.PHI_3
    else:
        return MessagesFormatterType.CHATML



def respond(
        message,
        history: list[tuple[str, str]],
        system_message,
        temperature,
        top_p,
        top_k,
        repetition_penalty,
):
    chat_template = get_messages_formatter_type("Mistral-7B-Instruct-v0.3-Q6_K.gguf")
    provider = LlamaCppServerProvider("http://localhost:8080")
    search_tool = WebSearchTool(
        llm_provider=provider,
        message_formatter_type=chat_template,
        model_max_context_tokens=get_context_by_model("Mistral-7B-Instruct-v0.3-Q6_K.gguf"),
        max_tokens_search_results=12000,
        max_tokens_per_summary=2048,
    )

    web_search_agent = LlamaCppAgent(
        provider,
        system_prompt=web_search_system_prompt,
        predefined_messages_formatter_type=chat_template,
        debug_output=True,
    )

    answer_agent = LlamaCppAgent(
        provider,
        system_prompt=system_message,
        predefined_messages_formatter_type=chat_template,
        debug_output=True,
    )

    settings = provider.get_provider_default_settings()
    settings.stream = False
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p

    settings.max_tokens = 2048
    settings.repeat_penalty = repetition_penalty

    output_settings = LlmStructuredOutputSettings.from_functions(
        [search_tool.get_tool()], add_thoughts_and_reasoning_field=True
    )

    messages = BasicChatHistory()

    for msn in history:
        user = {"role": Roles.user, "content": msn[0]}
        assistant = {"role": Roles.assistant, "content": msn[1]}
        messages.add_message(user)
        messages.add_message(assistant)

    result = web_search_agent.get_chat_response(
        f"Current Date and Time(d/m/y, h:m:s): {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}\n\nUser Query: " + message,
        llm_sampling_settings=settings,
        structured_output_settings=output_settings,
        add_message_to_chat_history=False,
        add_response_to_chat_history=False,
        print_output=False,
    )

    outputs = ""

    settings.stream = True
    response_text = answer_agent.get_chat_response(
        f"Write a detailed and complete research document that fulfills the following user request: '{message}', based on the information below.\n\n"
        + result[0]["return_value"],
        role=Roles.tool,
        llm_sampling_settings=settings,
        chat_history=messages,
        returns_streaming_generator=True,
        print_output=False,
    )

    for text in response_text:
        outputs += text
        yield outputs

    output_settings = LlmStructuredOutputSettings.from_pydantic_models(
        [CitingSources], LlmStructuredOutputType.object_instance
    )

    citing_sources = answer_agent.get_chat_response(
        "Cite the sources you used in your response.",
        role=Roles.tool,
        llm_sampling_settings=settings,
        chat_history=messages,
        returns_streaming_generator=False,
        structured_output_settings=output_settings,
        print_output=False,
    )
    outputs += "\n\nSources:\n"
    outputs += "\n".join(citing_sources.sources)
    yield outputs


# Begin Gradio UI
main = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(
            value=research_system_prompt,
            label="System message",
            interactive=True,
        ),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.45, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p",
        ),
        gr.Slider(
            minimum=0,
            maximum=100,
            value=40,
            step=1,
            label="Top-k",
        ),
        gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=1.1,
            step=0.1,
            label="Repetition penalty",
        ),
    ],
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="amber",
        neutral_hue="gray",
        font=[gr.themes.GoogleFont("Exo"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill_dark="#0c0505",
        block_background_fill_dark="#0c0505",
        block_border_width="1px",
        block_title_background_fill_dark="#1b0f0f",
        input_background_fill_dark="#140b0b",
        button_secondary_background_fill_dark="#140b0b",
        border_color_primary_dark="#1b0f0f",
        background_fill_secondary_dark="#0c0505",
        color_accent_soft_dark="transparent"
    ),
    css=css,
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
    submit_btn="Send",
    analytics_enabled=False,
    description="llama-cpp-agent research agent with ability to search the web",
    chatbot=gr.Chatbot(scale=1, placeholder=PLACEHOLDER),
)

if __name__ == "__main__":
    main.launch(server_name="0.0.0.0", server_port=7860)
