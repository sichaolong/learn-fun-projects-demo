# 使用 OpenAI Embedding
# from langchain.embeddings.openai import OpenAIEmbeddings
# 使用百度千帆 Embedding
import os

from dotenv import find_dotenv, load_dotenv
from langchain.embeddings.baidu_qianfan_endpoint import QianfanEmbeddingsEndpoint
# 使用我们自己封装的智谱 Embedding，需要将封装代码下载到本地使用
# from zhipuai_embedding import ZhipuAIEmbeddings

from langchain.vectorstores.chroma import Chroma
import re
from langchain.document_loaders.pdf import PyMuPDFLoader
''' 
* RecursiveCharacterTextSplitter 递归字符文本分割
RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
    这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
RecursiveCharacterTextSplitter需要关注的是4个参数：

* separators - 分隔符字符串数组
* chunk_size - 每个文档的字符数量限制
* chunk_overlap - 两份文档重叠区域的长度
* length_function - 长度计算函数
'''
# 导入文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter


# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
wenxin_api_key = os.environ["QIANFAN_ACCESS_KEY"]
wenxin_secret_key = os.environ["QIANFAN_SECRET_KEY"]

"""
数据清洗
"""
def data_cleaning(pdf_pages):

    for pdf_page in pdf_pages:
        # 匹配了一个前后不是中文字符的换行符。
        pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
        pdf_page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), pdf_page.page_content)
        pdf_page.page_content = pdf_page.page_content.replace('•', '')
        pdf_page.page_content = pdf_page.page_content.replace(' ', '')
        pdf_page.page_content = pdf_page.page_content.replace('\n\n', '\n')
    return pdf_pages



"""
数据分割
"""
def data_split(pdf_pages):
    # 知识库中单段文本长度
    CHUNK_SIZE = 500

    # 知识库中相邻文本重合长度
    OVERLAP_SIZE = 50

    # # 使用递归字符文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )
    # print(text_splitter.split_text(pdf_pages[1].page_content[0:1000]))

    split_docs = text_splitter.split_documents(pdf_pages)
    print(f"切分后的文件数量：{len(split_docs)}")
    print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")

    return split_docs


def data_process(path):
    loader = PyMuPDFLoader(path)
    # 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
    pdf_pages = loader.load()
    # 数据清洗
    cleaning_pdf_pages = data_cleaning(pdf_pages)
    # 数据分割
    split_pdf_pages = data_split(cleaning_pdf_pages)
    return split_pdf_pages


def data_presist_chroma(split_docs):
    # 定义 Embeddings
    # embedding = OpenAIEmbeddings()
    # embedding = ZhipuAIEmbeddings()
    embedding = QianfanEmbeddingsEndpoint()
    # 定义持久化路径
    persist_directory = '../vector_db/chroma'
    # 分割的数据
    vectordb = Chroma.from_documents(
        documents=split_docs[:20],  # 为了速度，只选择前 20 个切分的 doc 进行生成；使用千帆时因QPS限制，建议选择前 5 个doc
        embedding=embedding,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
    vectordb.persist()
    print(f"向量库中存储的数量：{vectordb._collection.count()}")

    question = "什么是四叉树"

    sim_docs = vectordb.similarity_search(question, k=3)
    print(f"检索到的内容数：{len(sim_docs)}")
    for i, sim_doc in enumerate(sim_docs):
        print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

    # 最大边际相关性 (MMR, Maximum marginal relevance) 检索
    mmr_docs = vectordb.max_marginal_relevance_search(question, k=3)
    for i, sim_doc in enumerate(mmr_docs):
        print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")


if __name__ == '__main__':
    path = "../asserts/搜索技术3-Lucene数据存储之BKD磁盘树.pdf"
    data = data_process(path)
    data_presist_chroma(data)









