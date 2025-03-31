from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix
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

def train_logit(X_train, y_train):
    """Trains logit model with cross-validation"""
    model_cv=LogisticRegressionCV(class_weight='balanced')
    model_cv.fit(X_train, y_train)
    return model_cv

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model on test dataset"""
    predictions = model.predict(X_test)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"Accuracy: {accuracy_score}")
    print(f"Confusion Matrix: {conf_matrix}")

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
            X_train, y_train = train_vectors.iloc[:, :-1], (train_vectors.iloc[:, -1]>6).astype(int)
            X_test, y_test = test_vectors.iloc[:, :-1], (test_vectors.iloc[:, -1]>6).astype(int)

            # Train logit model
            logit_model = train_logit(X_train, y_train)
            predictions = logit_model.predict(X_test)
            # Evaluate the model 
            balanced_accuracy = balanced_accuracy_score(y_test, predictions)
            print(f"Vector size: {vector_size}, Epochs: {epochs}, Balanced Accuracy: {balanced_accuracy}")
            with open('logit_res.json', 'a') as f: 
                    json.dump({"vector_size": vector_size, "epochs": epochs, "balanced_accuracy": balanced_accuracy}, f)
                    f.write('\n')
            # Track best balanced accuracy
            if balanced_accuracy > best_score: 
                best_score=balanced_accuracy
                best_params = (vector_size, epochs)

    print(f"Best model - vector size: {best_params[0]}, Epochs: {best_params[1]}, Balanced Accuracy: {best_score}")

if __name__ == '__main__': 
    main()