from embedding_llama_index import InstructorEmbeddings
# Import necessary classes of llama-index
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata


def ingest_docs_and_return_query_engine(path: str):
    # Setting the default llm of llama-index to None, llama-index will throw error otherwise!
    Settings.llm = None
    embed_model = InstructorEmbeddings(embed_batch_size=2)

    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    # load data
    user_docs = SimpleDirectoryReader(
        input_dir=path
    ).load_data()

    # build index
    user_index = VectorStoreIndex.from_documents(user_docs)

    user_query_engine = user_index.as_query_engine(similarity_top_k=5)

    user_query_engine_tool = QueryEngineTool(
        query_engine=user_query_engine,
        metadata=ToolMetadata(
            name="get_information_from_documents",
            description=(
                "Provides information from the documents uploaded by the user."
                "Use a detailed plain text question as input to the tool. It will retrieve the data by similarity search."
            ),
        ),
    )
    return user_query_engine_tool
