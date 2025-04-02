import pandas as pd
from bert.preprocessing import remove_emojis
import matplotlib.pyplot as plt
import re
def load_data(filepath): 
    data = pd.read_csv(filepath)
    return data

def clean_text(text):
    regex = r'[^а-яА-ЯёЁ\s]'
    return re.sub(regex, '', text)

def doc_length_hist(docs): 
    doc_lengths = [len(doc.split()) for doc in docs]
    doc_lengths = pd.Series(doc_lengths)
    ax = doc_lengths.hist(bins=50, color='skyblue', edgecolor='black', grid=False, figsize=(10, 6))
    plt.title('Number of words in a post')
    plt.xlabel('Number of words')
    plt.ylabel('Frequency')
    plt.show()

def doc_score_hist(scores): 
    doc_scores = pd.Series(scores)
    ax = doc_scores.hist(bins=15, color='skyblue', edgecolor='black', grid=False, figsize=(10, 6))
    plt.title('Number of posts per score')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.show()


def main(filepath): 
    data = load_data(filepath)
    docs = data['post']
    docs = docs.apply(clean_text)
    doc_lengths = doc_length_hist(docs)
    doc_scores = doc_score_hist(data['score'])
    print(pd.Series(doc_lengths).corr(data['score']))

