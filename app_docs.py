import gradio as gr
import os
import shutil

from llama_index.core.tools import ToolOutput
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from rag_pipeline import ingest_docs_and_return_query_engine
from agent import answer_agent
from llama_cpp_agent.chat_history import BasicChatHistory
from llama_cpp_agent.chat_history.messages import Roles

has_ingested = False
query_engine = None


def send_message(message):
    """
    Send a message to the user.
    Args:
        message (str): The message to send.
    Returns:
        none
    """
    return message


def ingest():
    global has_ingested, query_engine
    upload_folder = "uploaded_files"
    query_engine = ingest_docs_and_return_query_engine(upload_folder)
    has_ingested = True
    return "Documents ingested successfully. You can now start the chat."


# Function to handle chat messages
def chat_response(message, chat_history=[]):
    global has_ingested, query_engine
    if not has_ingested:
        return "Please ingest the files before!", chat_history

    history = BasicChatHistory()

    for msn in chat_history:
        user = {"role": Roles.user, "content": msn[0]}
        assistant = {"role": Roles.assistant, "content": msn[1]}
        history.add_message(user)
        history.add_message(assistant)

    structured_output_settings = LlmStructuredOutputSettings.from_llama_index_tools(
        [query_engine], add_thoughts_and_reasoning_field=True
    )
    print(structured_output_settings.get_gbnf_grammar())

    response = answer_agent.get_chat_response(message, chat_history=history, structured_output_settings=structured_output_settings)
    result_content = response[0]["return_value"].content if isinstance(response[0]["return_value"], ToolOutput) else response[0]["return_value"]

    result = answer_agent.get_chat_response(result_content, role=Roles.tool, chat_history=history)

    chat_history.append((message, result))

    return "", chat_history


# Function to handle file uploads
def upload_files(files):
    upload_folder = "uploaded_files"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    for file in files:
        shutil.copy(file.name, os.path.join(upload_folder, os.path.basename(file.name)))
    return f"{len(files)} file(s) uploaded successfully."


# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## Chat with Documents")

    with gr.Row():
        with gr.Column():
            chatbox = gr.Chatbot()

            chat_input = gr.Textbox(placeholder="Type your message here...")
            send_button = gr.Button("Send")
            ingest_documents = gr.Button("Ingest Documents and start Chat")

        with gr.Column():
            file_uploader = gr.File(label="Upload Files", file_count="multiple")
            upload_button = gr.Button("Upload")
            upload_output = gr.Textbox(interactive=False)

    # Set up the chat interaction
    send_button.click(chat_response, [chat_input, chatbox], [chat_input, chatbox])

    # Set up the ingest documents interaction
    ingest_documents.click(ingest, [], upload_output)

    # Set up the file upload interaction
    upload_button.click(upload_files, [file_uploader], upload_output)

# Launch the app
demo.launch()
