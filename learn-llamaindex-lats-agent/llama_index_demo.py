from llama_index.core.agent import AgentRunner

import qianfan
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.langchain import LangChainLLM
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings
from dotenv import load_dotenv
from wenxin_llm import Wenxin_LLM
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.storage import StorageContext
from transformers import AutoTokenizer, AutoModel

import os
import nest_asyncio


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['CURL_CA_BUNDLE'] = ''
nest_asyncio.apply()


def init_query_engine_tools():
    print("init_query_engine_tools ...")
    if not os.path.exists("./storage/高中英语试卷-useful-data"):
        # load data
        lats_docs = SimpleDirectoryReader(
            input_files=["./data/LATS论文.pdf"]
        ).load_data()
        paper_docs = SimpleDirectoryReader(
            input_files=["./data/高中英语试卷-useful-data.pdf"]
        ).load_data()

        print(lats_docs)
        print(paper_docs)

        # build index
        print("build index")
        lats_index = VectorStoreIndex.from_documents(lats_docs)
        paper_index = VectorStoreIndex.from_documents(paper_docs)

        # persist index
        print("persist index")
        lats_index.storage_context.persist(persist_dir="./storage/lats论文")
        paper_index.storage_context.persist(persist_dir="./storage/高中英语试卷-useful-data")
    else:
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/lats论文"
        )
        lats_index = load_index_from_storage(storage_context)

        storage_context = StorageContext.from_defaults(
            persist_dir="./storage/高中英语试卷-useful-data"
        )
        paper_index = load_index_from_storage(storage_context)

    lats_engine = lats_index.as_query_engine(similarity_top_k=3)
    paper_engine = paper_index.as_query_engine(similarity_top_k=3)

    print("prepare return query_engine_tools")
    query_engine_tools = [
        QueryEngineTool(
            query_engine=lats_engine,
            metadata=ToolMetadata(
                name="lats论文QueryEngineTool",
                description=(
                    "Provides information about LATS(Language Agent Tree Search) paper"
                    "Use a detailed plain text question as input to the tool. "
                    "The input is used to power a semantic search engine."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=paper_engine,
            metadata=ToolMetadata(
                name="辅助正确解答试题知识库QueryEngineTool",
                description=(
                    "知识库包含很多试题的信息供参考，试题内容包含题干、答案、解析信息 。 "
                    "使用详细的纯文本问题作为工具的输入。"
                    "输入用于为语义搜索引擎提供动力。"
                ),
            ),
        ),
    ]

    return query_engine_tools


if __name__ == '__main__':
    # NOTE: a higher temperate will help make the tree-expansion more diverse

    """
    1、OpenAI 需要代理
    """
    # llm = OpenAI(model="gpt-4o", temperature=0.6)

    """
    2、使用AzureOpenAI，M1芯片报错
    """
    # llm = AsyncAzureOpenAI(
    #     deployment_name="base4",
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    #     api_version="2023-07-01-preview",
    # )

    """
    3、使用HuggingFaceAPI，网络不稳定，需要代理
    """
    # llm = HuggingFaceInferenceAPI(
    #     model_name="bigscience/bloom",
    #     temperature=0.7,
    #     max_tokens=100,
    #     token=os.getenv("HUGGING_FACE_API_KEY"),  # Optional
    # )

    """
    4、使用LangChain LLM 包装 百度文新LLM
    """
    print("开始初始化模型...")
    llm = LangChainLLM(llm=Wenxin_LLM(model="Meta-Llama-3-8B",
                                      api_key=os.getenv("WEN_XIN_API_KEY"),
                                      secret_key=os.getenv("WEN_XIN_SECRET_KEY")))
    # response = llm.complete("什么是LATS（Language Agent Tree Search）？LATS和Agent有什么区别？使用Langchain4j如何构建LATS智能体？")
    # print("LLM直接回答：" + response.text)

    print("开始初始化embedding...")

    """
    5、OpenAIEmbedding需要代理
    """
    # embed_model = OpenAIEmbedding(model="text-embedding-3-small")


    """
    6、使用HuggingFace Embedding
    """
    # 如果使用报错找不到，使用huggingface模型 https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings/#modules
    embed_model = HuggingFaceEmbedding(model_name="./models/BAAI/bge-small-en-v1.5")
    # Load model directly
    # tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    # embed_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    # tokenizer.save_pretrained("./BAAI/bge-small-en-v1.5/tokenizer")
    # embed_model.save_pretrained("./BAAI/bge-small-en-v1.5/embedmodel")



    Settings.llm = llm
    Settings.embed_model = embed_model

    agent_worker = LATSAgentWorker.from_tools(
        init_query_engine_tools(),
        llm=llm,
        num_expansions=2,  # 指的是每个节点下可能探索的子操作的数量。 num_expansions=2 表示我们将探索每个父操作可能的下一步操作。
        max_rollouts=3,  # using -1 for unlimited rollouts,max_rollouts 指的是每次探索搜索空间的深度。 max_rollouts=5 表示在树中探索的最大深度为 5。
        verbose=True,
    )

    print("开始初始化agent...")
    # 官网文档有bug，https://github.com/run-llama/llama_index/issues/13140
    agent = AgentRunner(agent_worker=agent_worker)
    # task1
    task = agent.create_task(
        "解答试题，输出正确的答案与解析：_ good use you have made of your time to study, there is still room for improvement. A.Whatever B.However C.Though D.Whether"
    )
    # run initial step
    step_output = agent.run_step(task.task_id)
    print(step_output)
