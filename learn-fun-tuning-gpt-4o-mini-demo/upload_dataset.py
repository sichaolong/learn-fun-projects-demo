from openai import OpenAI

import os
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()

    # Get environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    print(client.files.create(
        file=open("assets/ai-answer-train-set-28-2803-10000.jsonl", "rb"),
        purpose="fine-tune"
    ))
    # FileObject(id='file-qOV5exrcXHIaLZkz2WFOmkLP', bytes=17383739, created_at=1722064665, filename='ai-answer-train-set-28-2803-10000.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)
