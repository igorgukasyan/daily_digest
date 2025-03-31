from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def load_data(filepath):
    with open(filepath, 'rb') as f: 
        data = pickle.load(f)
    return data

def train_elastic_net(X_train, y_train):
    """Trains elastic net model with cross-validation"""
    l1_ratio_vals = [x/100.0 for x in range (5, 100, 5)]
    model_cv=ElasticNetCV(l1_ratio=l1_ratio_vals)
    model_cv.fit(X_train, y_train)

    return model_cv

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """Evaluates the trained model on test dataset"""
    ## Predict on unseen
    predictions_test = model.predict(X_test)

    test_mse = mean_squared_error(y_test, predictions_test)
    test_mae = mean_absolute_error(y_test, predictions_test)
    test_r2 = model.score(X_test, y_test)

    print(f"Test MSE: {test_mse}")
    print(f"Test MAE: {test_mae}")
    print(f"Test R2 Score: {test_r2}")
    print('\n')
 
    ## Predict on train data
    predictions_train = model.predict(X_train)

    train_mse = mean_squared_error(y_train, predictions_train)
    train_mae = mean_absolute_error(y_train, predictions_train)
    train_r2 = model.score(X_train, y_train)

    print(f"Train MSE: {train_mse}")
    print(f"Train MAE: {train_mae}")
    print(f"Train R2 Score: {train_r2}")
    print('\n')

def main(filepath):
    data = load_data(filepath)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=7)

    # Extract    
    X_train, y_train = train_data.iloc[:, :-1], train_data['score'].values
    X_test, y_test = test_data.iloc[:, :-1], test_data['score'].values

    # Train elastic net model
    elastic_net_model = train_elastic_net(X_train, y_train)

    # Evaluate the model 
    evaluate_model(elastic_net_model, X_test, y_test, X_train, y_train)

if __name__ == '__main__': 
    main('../bert/embeddings.pickle')
    main('../bert/embeddings_without_emojis.pickle')