# Defaults
import datetime
import json

import PyPDF2
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from pdf2image import convert_from_path
from pydantic import BaseModel, Field
from pytesseract import pytesseract

from llama_cpp_agent.llm_prompt_template import PromptTemplate
# Locals
from llama_cpp_agent.providers import LlamaCppPythonProvider, LlamaCppServerProvider
from content import css, PLACEHOLDER
from llama_cpp_agent.tools import SummarizerTool
from llama_cpp_agent.tools.summarizing.tool import TextType
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
from llama_cpp_agent.tools import WebSearchTool, GoogleWebSearchProvider
from llama_cpp_agent.prompt_templates import web_search_system_prompt, research_system_prompt, \
    arxiv_search_system_prompt, general_information_assistant
import os
import requests

js_script = """
window.onload = function() {
    function disableUI() {
        document.querySelectorAll('input, button').forEach(function(item) {
            item.disabled = true;
        });
    }

    function enableUI() {
        document.querySelectorAll('input, button').forEach(function(item) {
            item.disabled = false;
        });
    }

    var form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(event) {
            event.preventDefault(); // Prevent form from submitting to observe the effect
            disableUI();
        });
    }
    
    window.addEventListener('gradio_event_prediction_complete', enableUI);
}
"""


class ArxivQuery(BaseModel):
    """
    A model representing search terms of an arxiv query.
    """
    search_terms: list[str] = Field(default_factory=list,
                                    description="Think step by step and identify important search terms based on the user request.")


def search_latest_arxiv_papers(query: str):
    """
    Searches the latest arxiv papers with a given query. Make sure to use a query effective for searching arxiv!
    Args:
        query (str): The query to search for
    Returns:
        (list): Returns a list of filenames of the downloaded arxiv papers
    :return:
    """
    folder_name = "temp_arxiv"
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&sortBy=relevance&sortOrder=descending&max_results={5}'
    response = requests.get(url)
    feed = response.content.decode('utf-8')
    papers = []
    for entry in feed.split('<entry>')[1:]:
        if '</entry>' in entry:
            id = entry.split('<id>')[1].split('<')[0].split('/')[-1]
            title = entry.split('<title>')[1].split('<')[0]
            papers.append((id, title))
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filenames = []
    for paper_id, paper_title in papers:
        pdf_url = f'https://arxiv.org/pdf/{paper_id}.pdf'
        pdf_filename = f'{folder_name}/{paper_id}.pdf'
        filenames.append(pdf_filename)
        with open(pdf_filename, 'wb') as f:
            f.write(requests.get(pdf_url).content)
        print(f'Downloaded {paper_title} ({paper_id})')
    return filenames


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


def ask_user(question: str):
    """
    Ask the user a question and return the answer.
    Args:
        question (str): The question to ask.
    Returns:
        (str): The answer.
    """
    return question


from PIL import Image
from joblib import Parallel, delayed
from pdf2image import convert_from_path
import pytesseract

# Preload pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your tesseract path


def process_page(page):
    # Convert the page to grayscale
    page = page.convert('L')

    # Apply OCR using the preloaded pytesseract
    page_text = pytesseract.image_to_string(page)

    return page_text


def process_pdf(path):
    pages = convert_from_path(path, dpi=300, fmt='PNG')
    page_texts = Parallel(n_jobs=-1)(delayed(process_page)(page) for page in pages)
    text = "\n".join(page_texts)
    return text


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

    web_search_agent = LlamaCppAgent(
        provider,
        system_prompt=PromptTemplate.from_string(arxiv_search_system_prompt).generate_prompt({"USER_QUERY": message}),
        predefined_messages_formatter_type=chat_template,
        debug_output=True,
    )

    answer_agent = LlamaCppAgent(
        provider,
        system_prompt=PromptTemplate.from_string(research_system_prompt).generate_prompt({"SUBJECT": message}),
        predefined_messages_formatter_type=chat_template,
        debug_output=True,
    )

    settings = provider.get_provider_default_settings()
    settings.stream = True
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p

    settings.max_tokens = 2048
    settings.repeat_penalty = repetition_penalty

    output_settings = LlmStructuredOutputSettings.from_pydantic_models(
        [ArxivQuery], LlmStructuredOutputType.object_instance
    )

    messages = BasicChatHistory()

    for msn in history:
        user = {"role": Roles.user, "content": msn[0]}
        assistant = {"role": Roles.assistant, "content": msn[1]}
        messages.add_message(user)
        messages.add_message(assistant)

    summarizer_tool = SummarizerTool(
        llm_provider=provider,
        message_formatter_type=chat_template,
        model_max_context_tokens=32768,
        max_tokens_per_summary=4096
    )

    result = web_search_agent.get_chat_response(
        f"Current Date and Time(d/m/y, h:m:s): {datetime.datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}",
        llm_sampling_settings=settings,
        structured_output_settings=output_settings,
        chat_history=messages,
        add_message_to_chat_history=False,
        add_response_to_chat_history=False,
        print_output=True,

    )
    paths = search_latest_arxiv_papers(" ".join(result.search_terms))
    pdf_texts = Parallel(n_jobs=-1)(delayed(process_pdf)(path) for path in paths)

    print("Start Summarizing")
    pdf_texts = summarizer_tool.summarize_text(message, pdf_texts, text_type=TextType.ocr)

    for idx, path in enumerate(paths):
        filename = os.path.basename(path)
        pdf_texts[idx] = f"File: {filename}\n\n" + pdf_texts[idx]

    yield '\n\n'.join(pdf_texts)

# Begin Gradio UI
main = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(
            value=research_system_prompt,
            label="System message",
            interactive=True,
        ),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.35, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.85,
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
