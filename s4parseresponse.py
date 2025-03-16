import json
import re
import s2clean
import numpy as np
with open('batch_67d61003821881908bb423c0d39cd054_output.jsonl', 'r') as f:
    responses = []
    for line in f: 
        responses.append(json.loads(line))
scores = []
for response in responses:
   scores.append(response.get('response', {}).get('body', {}).get('choices', {})[0].get('message', {}).get('content', {}))

coefs = {
    "Scale": 0.25,
    "Impact": 0.25,
    "Novelty": 0.05,
    "Potential": 0.25,
    "Legacy": 0.15,
    "Positivity": 0.05
}

def calculate_scores(scores):
    final_scores = []
    for score in scores: 
        all_scores = re.findall(r'\d', score)
        all_scores= [int(s) for s in all_scores]
        final_score = round(sum([coef * all_scores[i] for i, coef in enumerate(coefs.values())]),1)
        final_scores.append(final_score)
    return final_scores

final_scores = calculate_scores(scores)
