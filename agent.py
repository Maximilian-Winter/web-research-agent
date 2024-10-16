from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider

provider = LlamaCppServerProvider("http://localhost:8080")


answer_agent = LlamaCppAgent(
    provider,
    system_prompt="You are a helpful assistant.",
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL,
    debug_output=True,
)

search_pdf = """Your task is to use your 'get_information_from_documents' tool to retrieve information on the following user query:

<user_query>
{user_query}
</user_query>

Make sure you transform and optimize the user query for the input of the 'get_information_from_documents' tool, which will retrieve information based on similarity of the input.

After you retrieved the information, compose a clearly formatted answer on the user query."""