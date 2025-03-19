import json
import re
import s2clean as s2clean
import numpy as np
import pandas as pd
with open('./gpt_responses_4o_mini.json', 'r') as f:
    responses = json.load(f)

coefs = {
    "Scale": 0.25,
    "Impact": 0.25,
    "Novelty": 0.05,
    "Potential": 0.25,
    "Legacy": 0.15,
    "Positivity": 0.05
}

def calculate_scores(responses):
    final_scores = []
    for score in responses: 
        all_scores = re.findall(r'\d', score)
        all_scores= [int(s) for s in all_scores]
        final_score = round(sum([coef * all_scores[i] for i, coef in enumerate(coefs.values())]),1)
        final_scores.append(final_score)
    return final_scores

final_scores = calculate_scores(responses)
cleaned = s2clean.clean()
data = cleaned.iloc[:len(final_scores)]
data['score']=final_scores
data = data.iloc[:, [1,2]]
data = data[~data['post'].duplicated()]
data.to_csv('data.csv')