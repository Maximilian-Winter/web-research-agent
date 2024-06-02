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
    if not os.path.isdir(upload_folder) or not os.listdir(upload_folder):
        return "No uploaded files!"
    query_engine = ingest_docs_and_return_query_engine(upload_folder)
    has_ingested = True
    return "Documents pre-processed successfully. You can now start the chat."


# Function to handle chat messages
def chat_response(message, chat_history=[]):
    global has_ingested, query_engine
    if not has_ingested:
        return "Please pre-process the files before!", chat_history

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

    response = answer_agent.get_chat_response(message, chat_history=history,
                                              structured_output_settings=structured_output_settings)
    result_content = response[0]["return_value"].content if isinstance(response[0]["return_value"], ToolOutput) else response[0]["return_value"]

    result = answer_agent.get_chat_response(result_content, role=Roles.tool, chat_history=history)

    chat_history.append((message, result))

    return "", chat_history


# Function to handle file uploads
def upload_files(files):
    if files is None:
        return "No files to upload."
    upload_folder = "uploaded_files"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    for file in files:
        shutil.copy(file.name, os.path.join(upload_folder, os.path.basename(file.name)))
    return f"{len(files)} file(s) uploaded successfully."


def clear():
    global has_ingested, query_engine
    has_ingested = False
    upload_folder = "uploaded_files"
    os.remove(upload_folder)


# Define the Gradio interface with custom CSS
css = """
body {
    background-color: #121212;  /* Dark background for the entire body */
    body-background-fill: #121212
    color: #E0E0E0;  /* Light grey text color for readability */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;  /* Modern, readable font */
}

.block{
    background-color: #20262c;
}
.gradio-container {
    background-color: #121212;  /* Ensuring the container matches the body background */
    color: #E0E0E0;  /* Uniform text color throughout the app */
}

.gr-button {
    background-color: #333333;  /* Dark grey background for buttons */
    color: #FFFFFF;  /* White text for better contrast */
    border: 2px solid #555555;  /* Slightly lighter border for depth */
    padding: 10px 20px;  /* Adequate padding for touch targets */
    border-radius: 5px;  /* Rounded corners for modern feel */
    transition: background-color 0.3s;  /* Smooth transition for hover effect */
}

.gr-button:hover, .gr-button:active {
    background-color: #555555;  /* Lighter grey on hover/active for feedback */
    color: #FFFFFF;
}

.gr-textbox, .gr-markdown, .gr-chatbox, .gr-file, .gr-output-textbox {
    background-color: #2B2B2B;  /* Darker element backgrounds to distinguish from body */
    color: #E0E0E0;  /* Light grey text for readability */
    border: 1px solid #444444;  /* Slightly darker borders for subtle separation */
    border-radius: 5px;  /* Consistent rounded corners */
    padding: 10px;  /* Uniform padding for all input elements */
}

.gr-row {
    display: flex;
    justify-content: space-between;
    gap: 20px;  /* Adequate spacing between columns */
}

.gr-column {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;  /* Consistent gap between widgets within a column */
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## Chat with Documents")

    with gr.Row():
        with gr.Column():
            chatbox = gr.Chatbot()  # Initially hidden

            chat_input = gr.Textbox(label="Chat Input", placeholder="Type your message here..." )
            send_button = gr.Button("Send")
        with gr.Column():
            status_output = gr.Textbox(label="Document Status", interactive=False, placeholder="Please upload your documents and pre-process them below" )
            file_uploader = gr.File(label="Upload Files", file_count="multiple")
            upload_button = gr.Button("Upload")
            ingest_documents = gr.Button("Preprocess documents")
            clear_button = gr.Button("Delete uploaded documents")

    # Chat response function (omitted for brevity, remains unchanged)
    send_button.click(chat_response, [chat_input, chatbox], [chat_input, chatbox])

    # Set up the ingest documents interaction
    ingest_documents.click(ingest, [], status_output)
    clear_button.click(clear)
    # Set up the file upload interaction
    upload_button.click(upload_files, [file_uploader], status_output)

# Launch the app
demo.launch()
