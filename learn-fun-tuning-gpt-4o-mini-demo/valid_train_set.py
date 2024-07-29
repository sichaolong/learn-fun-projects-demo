import json
import tiktoken # for token counting
import numpy as np
from collections import defaultdict

encoding = tiktoken.get_encoding("cl100k_base")


def load_json_data():
    data_path = "assets/ai-answer-train-set-28-2803-10000.jsonl"

    # Load the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)
    return dataset

def valid_json_data(dataset):
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

# not exact!
# simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def num_tokens_from_dataset(dataset):
    num_tokens = 0
    for ex in dataset:
        messages = ex["messages"]
        for message in messages:
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def num_assistant_tokens_from_dataset(dataset):
    num_tokens = 0
    for ex in dataset:
        messages = ex["messages"]
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
    return num_tokens




if __name__ == '__main__':
    dataset = load_json_data()
    valid_json_data(dataset)

    print("num_tokens_from_dataset:" ,num_tokens_from_dataset(dataset))
    print("num_assistant_tokens_from_dataset:", num_assistant_tokens_from_dataset(dataset))

