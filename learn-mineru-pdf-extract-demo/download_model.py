# pip install modelscope

# 使用modelscope sdk下载模型
from modelscope import snapshot_download

if __name__ == '__main__':
    model_dir = snapshot_download('wanderkid/PDF-Extract-Kit')
    print(model_dir)
    # /Users/sichaolong/.cache/modelscope/hub/wanderkid/PDF-Extract-Kit