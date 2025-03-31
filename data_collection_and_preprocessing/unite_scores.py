import json
def unite_scorings():
    with open('../data/gpt_responses_4o_mini.json', 'r') as f:
        responses = json.load(f)
    with open('../data/gpt_responses_4o_mini_2.json', 'r') as f: 
        responses_2 = json.load(f)
    with open('../data/gpt_responses_4o_mini_3.json', 'r') as f: 
        responses_3 = json.load(f)
    with open('../data/gpt_responses_4o_mini_4.json', 'r') as f: 
        responses_4 = json.load(f)
    with open('../data/gpt_responses_4o_mini_5.json', 'r') as f: 
        responses_5 = json.load(f)
    responses_total = responses+responses_2+responses_3+responses_4+responses_5
    return responses_total