import os

from dotenv import load_dotenv
from openai import OpenAI

def train_task_info():
    from openai import OpenAI
    client = OpenAI()

    # List 10 fine-tuning jobs
    client.fine_tuning.jobs.list(limit=10)

    # Retrieve the state of a fine-tune
    client.fine_tuning.jobs.retrieve("ftjob-abc123")

    # Cancel a job
    client.fine_tuning.jobs.cancel("ftjob-abc123")

    # List up to 10 events from a fine-tuning job
    client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)

    # Delete a fine-tuned model (must be an owner of the org the model was created in)
    client.models.delete("ft:gpt-3.5-turbo:acemeco:suffix:abc123")

if __name__ == '__main__':
    load_dotenv()

    file_id = 'file-qOV5exrcXHIaLZkz2WFOmkLP'
    # Get environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18"
    )

    print("启动微调任务：",job)

    jobs = client.fine_tuning.jobs.list(limit=10)
    print("当前微调任务列表：",jobs)
    job_events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=jobs[0], limit=10)
    print("微调任务事件：",job_events)


