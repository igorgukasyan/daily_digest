import s2clean
import json
from openai import OpenAI
import os

cleaned = s2clean.clean()
prompt = """\n### Instruction:\nAnalyze the following Russian news snippet and provide scores for each of the following dimensions. Please respond in the format: \"Dimension: Score\" (e.g., \"Scale: 8\"). Ensure your response is concise and only includes the dimension name followed by a score.\n\n### Dimensions:\n1. **Scale**: How broadly the event affects humanity.\n2. **Impact**: How strong the immediate effect is.\n3. **Novelty**: How unique and unexpected is the event.\n4. **Potential**: How likely it is to shape the future.\n5. **Legacy**: How likely it is to be considered a turning point in history or a major milestone.\n6. **Positivity**: How positive is the event.\n\n### Input:\n[Insert Russian news snippet here]\n\n### Response Format:\nScale: [Score]  \nImpact: [Score]  \nNovelty: [Score]  \nPotential: [Score]  \nLegacy: [Score]  \nPositivity: [Score]\n"""

def generate_jsonl(cleaned, num_requests):
    with open('batch_requests_evaluation.jsonl', 'w') as file:
        for i in range(1, num_requests+1):
            request_data = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "developer", "content": prompt},
                        {"role": "user", "content": cleaned['post'][i-1]}
                    ],
                    "max_completion_tokens": 33
                }
            }
            json.dump(request_data, file, ensure_ascii=False)
            file.write('\n')

generate_jsonl(cleaned, 200)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def upload_batch_input_file(file_name):
    batch_input_file = client.files.create(
        file=open(file_name, "rb"),
        purpose="batch"
    )
    return batch_input_file


batch_input_file = upload_batch_input_file("batch_requests_evaluation.jsonl")

batch_input_file_id = batch_input_file.id
client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)

    