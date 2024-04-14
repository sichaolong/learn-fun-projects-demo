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


if __name__ == '__main__':
    # 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
    loader = PyMuPDFLoader("../asserts/搜索技术3-Lucene数据存储之BKD磁盘树.pdf")

    # 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
    pdf_pages = loader.load()

    print(f"载入后的变量类型为：{type(pdf_pages)}，", f"该 PDF 一共包含 {len(pdf_pages)} 页")
    # 载入后的变量类型为：<class 'list'>， 该 PDF 一共包含 20 页
    pdf_page = pdf_pages[0]

    print(f"每一个元素的类型：{type(pdf_page)}.",
          f"该文档的描述性数据：{pdf_page.metadata}",
          f"查看该文档的内容:\n{pdf_page.page_content}",
          sep="\n------\n")

    print("============================================")

    # re数据清洗，去掉换行以及空格
    cleaning_pdf_pages = data_cleaning(pdf_pages)
    print(f"清洗之后每一个元素的类型：{type(cleaning_pdf_pages[0])}.",
          f"清洗之后该文档的描述性数据：{cleaning_pdf_pages[0].metadata}",
          f"清洗之后查看该文档的内容:\n{cleaning_pdf_pages[0].page_content}",
          sep="\n------\n")

    # 数据分割
    split_pdf_pages = data_split(cleaning_pdf_pages)
    print(f"清洗之后每一个元素的类型：{type(cleaning_pdf_pages[0])}.",
          f"清洗之后该文档的描述性数据：{cleaning_pdf_pages[0].metadata}",
          f"清洗之后查看该文档的内容:\n{cleaning_pdf_pages[0].page_content}",
          sep="\n------\n")
