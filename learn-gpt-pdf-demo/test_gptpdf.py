import os

from GeneralAgent import Agent
from gptpdf import parse_pdf

# laod environment variables from .env file
import dotenv

dotenv.load_dotenv()

pdf_path = './examples/[3535413515862016]-甘肃省陇南市成县2023-2024学年八年级下学期质量监测物理试卷（四）.pdf'
output_dir = './examples/[3535413515862016]-甘肃省陇南市成县2023-2024学年八年级下学期质量监测物理试卷（四）/'

# 清空output_dir
import shutil

shutil.rmtree(output_dir, ignore_errors=True)


def test_use_api_key():
    api_key = os.getenv('OPENAI_API_KEY')
    base_url = os.getenv('OPENAI_API_BASE')
    # Manually provide OPENAI_API_KEY and OPEN_API_BASE
    content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, api_key=api_key, base_url=base_url,
                                     model='gpt-4o', gpt_worker=6)
    print(content)
    print(image_paths)
    # also output_dir/output.md is generated


def test_use_env():

    # Use OPENAI_API_KEY and OPENAI_API_BASE from environment variables
    content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, model='gpt-4o', verbose=True)
    print(content)
    print(image_paths)
    # also output_dir/output.md is generated


def test_azure():
    api_key = '8d728a3da84146a7a55396a7e8abb3ea'  # Azure API Key
    base_url = 'https://xkwopenai.openai.azure.com/'  # Azure API Base URL
    model = 'azure_base4o'  # azure_ with deploy ID name (not open ai model name), e.g. azure_cpgpt4
    # Use OPENAI_API_KEY and OPENAI_API_BASE from environment variables
    content, image_paths = parse_pdf(pdf_path, output_dir=output_dir, api_key=api_key, base_url=base_url, model=model,
                                     verbose=True)
    print(content)
    print(image_paths)

def test_agent():
    api_key = '8d728a3da84146a7a55396a7e8abb3ea'  # Azure API Key
    base_url = 'https://xkwopenai.openai.azure.com'  # Azure API Base URL
    model = 'azure_base4'  # azure_ with deploy ID name (not open ai model name), e.g. azure_cpgpt4
    agent = Agent(api_key=api_key, base_url=base_url, model=model, disable_python_run=True)
    print(agent.run("你好"))

if __name__ == '__main__':
    # test_agent()

    # test_use_api_key()
    # test_use_env()
    test_azure()
