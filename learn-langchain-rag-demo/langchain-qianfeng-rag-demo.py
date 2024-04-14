# 读取本地/项目的环境变量。
import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms.baidu_qianfan_endpoint import QianfanLLMEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
wenxin_api_key = os.environ["QIANFAN_ACCESS_KEY"]
wenxin_secret_key = os.environ["QIANFAN_SECRET_KEY"]

# 向量数据库持久化路径
persist_directory = './vector_db/chroma'



"""
加载Chroma向量数据库
"""
def init_chroma_db(path):
    embedding = QianfanEmbeddingsEndpoint()
    # 加载数据库
    vectordb = Chroma(
        persist_directory=path,  # 允许我们将persist_directory目录保存到磁盘上
        embedding_function=embedding
    )
    print(f"向量库中存储的数量：{vectordb._collection.count()}")
    return vectordb



"""
相似性搜索
"""
def similarity_search_from_chroma(vectordb,ques,k=3):
    docs = vectordb.similarity_search(ques, k=k)
    print(f"检索到的内容数：{len(docs)}")
    for i, doc in enumerate(docs):
        print(f"检索到的第{i}个内容: \n {doc.page_content}",end="\n-----------------------------------------------------\n")
    return docs



"""
构建RAG检索链
"""
def rag_chain(template,vectordb,question):
    llm = QianfanLLMEndpoint(streaming=True)
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                     template=template)


    # 借助langchain将对话上下文存储起来
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    #     return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    # )


    """
    指定 llm：指定使用的 LLM
    指定 chain type : RetrievalQA.from_chain_type(chain_type="map_reduce")，也可以利用load_qa_chain()方法指定chain type。
    自定义 prompt ：通过在RetrievalQA.from_chain_type()方法中，指定chain_type_kwargs参数，而该参数：chain_type_kwargs = {"prompt": PROMPT}
    返回源文档：通过RetrievalQA.from_chain_type()方法中指定：return_source_documents=True参数；也可以使用RetrievalQAWithSourceChain()方法，返回源文档的引用（坐标或者叫主键、索引）
    """
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=vectordb.as_retriever(),
                                           return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


    # 大模型自己回答的效果
    prompt_template = """请回答下列问题:{}""".format(question)
    print("(1)大模型回答 question 的结果：")
    print(llm.predict(prompt_template))


    # rag + llm回答的效果
    result = qa_chain({"query": question})
    print("(2)大模型+知识库后回答 question 的结果：")
    print(result["result"])

if __name__ == '__main__':

    # 初始化数据库
    path = persist_directory
    chroma_vectordb = init_chroma_db(path)

    # 搜索向量数据库
    # question = "什么是BKD-tree?"
    # result_docs = similarity_search_from_chroma(chroma_vectordb,question,3)

    # 构建检索问答链
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    """

    question = "为什莫要求t = Θ(min(M/B, √M))？"
    rag_chain(template,chroma_vectordb,question)


    """
    向量库中存储的数量：20
    
    大模型回答 question 的结果：
    要求t = Θ(min(M/B, √M))的原因如下：
    
    1. 最小值优化：当M和B之间的比例较小（即M/B较小）时，t的值将受到M/B的影响更大。因此，使用min(M/B, √M)作为t的上界可以确保在M/B较小时能够得到更准确的估计。
    2. 性能优化：对于较小的M和B，我们通常希望算法的复杂度或时间复杂度尽可能低。因此，t的上界应基于最小可能的计算时间来选择。在这种情况下，min(M/B, √M)将确保t具有更低的上限，从而有助于性能优化。
    总之，要求t = Θ(min(M/B, √M))是基于最小值优化的原则，通过限制t的上界为较小的数值，可以提高算法的性能并降低其复杂度。这使得在较小输入条件下能够得到更准确的估计，并且对更大的输入也不会产生过度复杂或耗费时间的结果。
    
    大模型+知识库后回答 question 的结果：
    要求t = Θ(min(M/B, √M))是因为KD-Tree需要构建一个二叉树结构，树的深度通常受限于数据集的大小M和节点数B之间的最小值。为了保持树的平衡性和减少树的深度，通常需要限制节点的最大深度，即要求t = Θ(min(M/B, √M))。这样可以保证树的结构更加稳定，并且能够更有效地进行最邻近搜索。
    
    """


    question = "对于大规模多维度数据近似搜索,Lucene采用什么结构？"
    rag_chain(template,chroma_vectordb,question)


    """
    (1)大模型回答 question 的结果：
    对于大规模多维度数据近似搜索，Lucene通常采用倒排索引（Inverted Index）结构。
    倒排索引是一种将文档中的词汇转换为索引的数据结构，它能够快速地查找与某个词汇相关的文档。对于大规模多维度数据近似搜索，Lucene通过将文档和词汇按照多维空间进行组织，建立相应的倒排索引，从而实现对大规模多维度数据的快速搜索。
    具体而言，Lucene使用倒排索引来存储每个文档中出现的词汇及其在文档中的位置信息，以及这些词汇在查询中出现的频率等信息。通过这种方式，当进行搜索时，Lucene可以快速地在倒排索引中查找与查询条件匹配的文档，从而实现对大规模多维度数据的近似搜索。
    
    
    (2)大模型+知识库后回答 question 的结果：
    Lucene采用BKD-Tree结构来满足大规模多维度数据的近似搜索。
    """