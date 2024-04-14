import os

import qianfan
from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
wenxin_api_key = os.environ["QIANFAN_ACCESS_KEY"]
wenxin_secret_key = os.environ["QIANFAN_SECRET_KEY"]


def wenxin_embedding(text: str):
    emb = qianfan.Embedding()

    """
        模型名称。说明：
    （1）不填写此参数，默认模型为Embedding-V1
    （2） 如果需指定支持预置服务的模型，此字段必填，支持模型如下：
    · Embedding-V1
    · bge-large-en
    · bge-large-zh
    · tao-8k
    """
    resp = emb.do(model="bge-large-zh", texts=[  # 非默认模型，需填写 model参数
        "世界上最高的山"
    ])
    return resp['data'][0]['embedding']


if __name__ == '__main__':
    text = "要生成 embedding 的输入文本，字符串形式。"
    response = wenxin_embedding(text=text)
    print(response)
