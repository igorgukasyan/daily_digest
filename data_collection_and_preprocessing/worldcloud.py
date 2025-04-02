import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def load_data(filepath): 
    data = pd.read_csv(filepath)
    return data

def combine_docs(corpus): 
    combined = " ".join(corpus['post'])
    return combined

def create_wordcloud(corpus): 
    combined = combine_docs(corpus)
    wordcloud = WordCloud(
        background_color= "white",
        width = 800,
        height = 600,
        stopwords={'это'}
    ).generate(combined)
    return wordcloud

def main():
    data = load_data('./data/data_preprocessed.csv')
    wordcloud = create_wordcloud(data)
    plt.figure(figsize=(10,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    wordcloud.to_file("./portfolio_assets/wordcloud.png")
    

if __name__ == '__main__':
    main()