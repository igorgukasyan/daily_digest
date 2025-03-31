import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
from tqdm import tqdm

def load_data(filepath): 
    data = pd.read_csv(filepath).iloc[:, 1:]
    return data

def lemmatize_corpus():
    data=load_data('data.csv')
    nlp = spacy.load('ru_core_news_md', disable = ['parser','ner'])
    
    lemmatized_posts= []
    for post in tqdm(data['post'], desc='Lemmatizing posts'): 
        doc = nlp(post)
        lemmatized_posts.append(" ".join([token.lemma_ for token in doc]))

    data['post'] = lemmatized_posts
    return data

def remove_stop_words_and_short_docs(df, min_doc_length=0, language="russian"):
    """Removes stop words and short documents."""
    stop_words = set(stopwords.words(language))

    num_words = []
    cleaned_posts = []

    ## Remove stop words
    for post in df['post']:
        words = word_tokenize(post, language=language)
        filtered_post = [word for word in words if word.lower() not in stop_words]
        
        ## Track number of words in each post
        num_words.append(len(filtered_post))
        cleaned_posts.append(" ".join(filtered_post))

    df['post'] = cleaned_posts
    df['word_count'] = num_words

    df = df[df['word_count']>min_doc_length].copy()

    return df

if __name__ == "__main__": 
    data = lemmatize_corpus()
    data.to_csv('lemmatized_data.csv', index=False)
    data = remove_stop_words_and_short_docs(data, min_doc_length=5)
    data.to_csv('preprocessed_data.csv', index=False)