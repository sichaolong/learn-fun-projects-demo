import qianfan

from dotenv import find_dotenv, load_dotenv
import os

# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
wenxin_api_key = os.environ["QIANFAN_ACCESS_KEY"]
wenxin_secret_key = os.environ["QIANFAN_SECRET_KEY"]

def gen_wenxin_messages(prompt):
    '''
    构造文心模型请求参数 messages

    请求参数：
        prompt: 对应的用户提示词
    '''
    messages = [{"role": "user", "content": prompt}]
    return messages


def get_completion(prompt, model="ERNIE-Bot", temperature=0.01):
    """

        模型名称，用于指定平台支持预置服务的模型，说明：
    （1）不填写此字段时，默认模型为ERNIE-Bot-turbo
    （2）如果模型为平台支持预置服务的模型，此字段必填，且支持模型名称如下：
    · ERNIE-Bot-4：模型版本为ERNIE 4.0
    · ERNIE-Bot：模型版本为ERNIE-3.5-8K
    · ERNIE-3.5-4K-0205：模型版本为ERNIE-3.5-4K-0205
    · ERNIE-3.5-8K-0205：模型版本为ERNIE-3.5-8K-0205
    · ERNIE-3.5-8K-1222：模型版本为ERNIE-3.5-8K-1222
    · ERNIE-Speed：模型版本为ERNIE-Speed-8K
    · ERNIE-Speed-128k
    · ERNIE-Bot-turbo：模型为ERNIE Lite
    · ERNIE-Lite-8K-0308
    · ERNIE-Tiny-8K
    · EB-turbo-AppBuilder：模型为ERNIE-Speed-AppBuilder
    · Gemma-7B-it
    · Yi-34B-Chat
    · BLOOMZ-7B
    · Qianfan-BLOOMZ-7B-compressed
    · Mixtral-8x7B-Instruct
    · Llama-2-7b-chat
    · Llama-2-13b-chat
    · Llama-2-70b-chat
    · Qianfan-Chinese-Llama-2-7B
    · Qianfan-Chinese-Llama-2-13B：模型版本为Qianfan-Chinese-Llama-2-13B-v1
    · ChatGLM2-6B-32K
    · XuanYuan-70B-Chat-4bit
    · ChatLaw
    · AquilaChat-7B
    """
    '''
    获取文心模型调用结果

    请求参数：
        prompt: 对应的提示词
        model: 调用的模型，默认为 ERNIE-Bot，也可以按需选择 ERNIE-Bot-4 等其他模型
        temperature: 模型输出的温度系数，控制输出的随机程度，取值范围是 0~1.0，且不能设置为 0。温度系数越低，输出内容越一致。
    '''

    chat_comp = qianfan.ChatCompletion()
    message = gen_wenxin_messages(prompt)

    resp = chat_comp.do(messages=message,
                        model=model,
                        temperature=temperature,
                        system="你是一名个人助理-小鲸鱼")

    return resp["result"]


if __name__ == '__main__':
    print(get_completion("你好，介绍一下你自己"))
