from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from dotenv import load_dotenv


import os
import nest_asyncio

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.storage import StorageContext

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
nest_asyncio.apply()

if not os.path.exists("./storage/lyft"):
    # load data
    lyft_docs = SimpleDirectoryReader(
        input_files=["./data/10k/lyft_2021.pdf"]
    ).load_data()
    uber_docs = SimpleDirectoryReader(
        input_files=["./data/10k/uber_2021.pdf"]
    ).load_data()

    # build index
    lyft_index = VectorStoreIndex.from_documents(lyft_docs)
    uber_index = VectorStoreIndex.from_documents(uber_docs)

    # persist index
    lyft_index.storage_context.persist(persist_dir="./storage/lyft")
    uber_index.storage_context.persist(persist_dir="./storage/uber")
else:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/lyft"
    )
    lyft_index = load_index_from_storage(storage_context)

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/uber"
    )
    uber_index = load_index_from_storage(storage_context)

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool. "
                "The input is used to power a semantic search engine."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool. "
                "The input is used to power a semantic search engine."
            ),
        ),
    ),
]

if __name__ == '__main__':
    # NOTE: a higher temperate will help make the tree-expansion more diverse
    llm = OpenAI(model="gpt-4o", temperature=0.6)
    # 使用报错找不到，使用huggingface模型 https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/#modules
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    agent_worker = LATSAgentWorker.from_tools(
        query_engine_tools,
        llm=llm,
        num_expansions=2,  # 指的是每个节点下可能探索的子操作的数量。 num_expansions=2 表示我们将探索每个父操作可能的下一步操作。
        max_rollouts=3,  # using -1 for unlimited rollouts,max_rollouts 指的是每次探索搜索空间的深度。 max_rollouts=5 表示在树中探索的最大深度为 5。
        verbose=True,
    )
    agent = agent_worker.as_worker()

    # task1
    task = agent.create_task(
        "Given the risk factors of Uber and Lyft described in their 10K files, "
        "which company is performing better? Please use concrete numbers to inform your decision."
    )
    # run initial step
    step_output = agent.run_step(task.task_id)
    print(step_output)