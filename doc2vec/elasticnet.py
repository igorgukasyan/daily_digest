from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd
import gensim
from sklearn.model_selection import train_test_split
import json

def load_data(filepath): 
    data = pd.read_csv(filepath).iloc[:, :-1]
    return data

def read_corpus(series, tokens_only = False):
    """Process text into tokens form for Doc2Vec"""
    for index, line in enumerate(series): 
        tokens = gensim.utils.simple_preprocess(line)
        if tokens_only: 
            yield tokens 
        else: 
            yield gensim.models.doc2vec.TaggedDocument(tokens, [index])

def train_doc2vec_model(corpus, vector_size, epochs, min_count):
    """"Trains a Doc2Vec model on the provided corpus"""
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, epochs = epochs, min_count =min_count)
    ## vocabulary is essentially a list of all unique words extracte from the corpus
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def extract_vectors(model, test_corpus, train_data, test_data):
    """Extract vetors and prepare train/test datasets"""
    train_vectors = [model.dv[i] for i in range(len(model.dv))]
    train_vectors_df = pd.DataFrame(train_vectors)
    train_vectors_df['score']=train_data['score'].reset_index(drop=True)
    
    test_vectors = [model.infer_vector(doc) for doc in test_corpus]
    test_vectors_df = pd.DataFrame(test_vectors)
    test_vectors_df['score']=test_data['score'].reset_index(drop=True)
    return train_vectors_df, test_vectors_df

def train_elastic_net(X_train, y_train):
    """Trains elastic net model with cross-validation"""
    l1_ratio_vals = [x/100.0 for x in range (5, 100, 5)]
    model_cv=ElasticNetCV(l1_ratio=l1_ratio_vals)
    model_cv.fit(X_train, y_train)

    model_final = ElasticNet(l1_ratio=model_cv.l1_ratio_, alpha=model_cv.alpha_)
    model_final.fit(X_train, y_train)
    return model_final

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model on test dataset"""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = model.score(X_test, y_test)

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R2 Score: {r2}")

def main():
    data = load_data('../data/data_preprocessed.csv')
    train_data, test_data = train_test_split(data, test_size=0.2)

    # Pre-process and tokenize
    train_corpus = list(read_corpus(train_data.iloc[:, 0]))
    test_corpus = list(read_corpus(test_data.iloc[:, 0], tokens_only=True))

    best_score = float('-inf')
    best_params = None
    
    # Find best parameter values
    for vector_size in [10, 30, 50, 100, 150]:
        for epochs in [10, 30, 40, 60]:
            doc2vec_model = train_doc2vec_model(train_corpus,
                                                 vector_size=vector_size,
                                                 epochs=epochs,
                                                 min_count = 2)

            # Extract vectors
            train_vectors, test_vectors = extract_vectors(doc2vec_model, test_corpus, train_data, test_data)
            X_train, y_train = train_vectors.iloc[:, :-1], train_vectors.iloc[:, -1]
            X_test, y_test = test_vectors.iloc[:, :-1], test_vectors.iloc[:, -1]

            # Train elastic net model
            elastic_net_model = train_elastic_net(X_train, y_train)

            # Evaluate the model 
            r2_score = elastic_net_model.score(X_test, y_test)
            print(f"Vector size: {vector_size}, Epochs: {epochs}, R2: {r2_score}")
            with open('elasticnet_res.json', 'a') as f: 
                json.dump({"vector_size": vector_size, "epochs": epochs, "r2": r2_score}, f)
                f.write('\n')
            # Track best R2
            if r2_score > best_score: 
                best_score=r2_score
                best_params = (vector_size, epochs)

    print(f"Best model - vector size: {best_params[0]}, Epochs: {best_params[1]}, R2: {best_score}")

if __name__ == '__main__': 
    main()