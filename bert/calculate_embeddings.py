from sentence_transformers import SentenceTransformer
from preprocessing import remove_emojis
import pandas as pd
import pickle

def load_data(filepath): 
    data = pd.read_csv(filepath)
    return data

def calculate_embeddings(model, data):
    """Calculate embeddings on posts as they (almost) are in TG"""
    vectors = model.encode(data['post'].tolist(), batch_size = 32)
    dataframe = pd.DataFrame(vectors)
    dataframe['score']=data['score'].reset_index(drop=True)
    return dataframe

def calculate_embeddings_without_emojis(model, data):
    """Calculate embeddings on posts with no emojis"""
    data['post']=data['post'].apply(remove_emojis)
    vectors = model.encode(data['post'].tolist(), batch_size = 32)
    dataframe = pd.DataFrame(vectors)
    dataframe['score']=data['score'].reset_index(drop=True)
    return dataframe

def main():
    model = SentenceTransformer('sergeyzh/rubert-tiny-turbo')
    data = load_data('../data/data.csv').iloc[:, 1:3]

    embeddings = calculate_embeddings(model, data)
    with open('embeddings.pickle', "wb") as f: 
        pickle.dump(embeddings, f) 
    embeddings_without_emojis = calculate_embeddings_without_emojis(model, data)
    with open('embeddings_without_emojis.pickle', 'wb') as f: 
        pickle.dump(embeddings_without_emojis, f)

if __name__ == '__main__': 
    main()