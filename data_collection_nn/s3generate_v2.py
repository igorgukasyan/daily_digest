import s2clean
import json
from openai import OpenAI
import os
import pandas as pd
import aiofiles
cleaned = s2clean.clean()
dev_prompt = """\n### Instruction:\nAnalyze the following Russian news snippet and provide scores for each of the following dimensions. Please respond in the format: \"Dimension: Score\" (e.g., \"Scale: 8\"). Ensure your response is concise and only includes the dimension name followed by a score.\n\n### Dimensions:\n1. **Scale**: How broadly the event affects humanity.\n2. **Impact**: How strong the immediate effect is.\n3. **Novelty**: How unique and unexpected is the event.\n4. **Potential**: How likely it is to shape the future.\n5. **Legacy**: How likely it is to be considered a turning point in history or a major milestone.\n6. **Positivity**: How positive is the event.\n\n### Input:\n[Insert Russian news snippet here]\n\n### Response Format:\nScale: [Score]  \nImpact: [Score]  \nNovelty: [Score]  \nPotential: [Score]  \nLegacy: [Score]  \nPositivity: [Score]\n"""

client = OpenAI()
def generate_gpt_responses(cleaned):
    try: 
        gpt_responses = []
        output_file = 'gpt_responses_4o_mini.json'
        prompts = list(cleaned.iloc[:, 1])
        
        for i, prompt in enumerate(prompts):
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "developer", "content": dev_prompt},
                    {"role": "user", "content": prompts[i]}
                ],
                max_completion_tokens=33
            )
            response = completion.choices[0].message.content
            gpt_responses.append(response)
            print(f'Done with response: {i + 1}')

            if (i + 1) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(gpt_responses, f, indent=4)
                print(f'Saved checkpoint at {i + 1}')

        # Final save after loop completes        
        with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            json.dump(gpt_responses, f, indent=4)
        print('Final data saved.')

        return cleaned
    except Exception as e:
        print(f"Error: {e}")
        return {"gpt_responses": gpt_responses, "cleaned": cleaned}