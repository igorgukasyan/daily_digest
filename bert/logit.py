from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def load_data(filepath):
    with open(filepath, 'rb') as f: 
        data = pickle.load(f)
    return data

def train_logit(X_train, y_train):
    """Trains logit model with cross-validation"""
    model_cv=LogisticRegressionCV(class_weight='balanced')
    model_cv.fit(X_train, y_train)
    return model_cv

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluates the trained model on both training and test datasets and prints results in a table."""
    
    def evaluate_split(data_split_name, X, y):
        """Helper function to evaluate a specific data split."""
        predictions = model.predict(X)
        balanced_accuracy = balanced_accuracy_score(y, predictions)
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)
        
        # Display confusion matrix
        ConfusionMatrixDisplay.from_predictions(y, predictions)
        
        return {
            "Dataset": data_split_name,
            "Balanced Accuracy": balanced_accuracy,
            "Accuracy": accuracy,
            "F1 Score": f1
        }
    
    # Evaluate on training and test data
    train_results = evaluate_split("Train", X_train, y_train)
    test_results = evaluate_split("Test", X_test, y_test)
    
    # Create a DataFrame for tabular display
    results_df = pd.DataFrame([train_results, test_results])
    
    # Print the results as a table
    print("\nEvaluation Results:")
    print(results_df.to_string(index=False))

def main(filepath):
    data = load_data(filepath)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=7)

    # Extract    
    X_train, y_train = train_data.iloc[:, :-1], (train_data['score']>6).astype(int)
    X_test, y_test = test_data.iloc[:, :-1], (test_data['score']>6).astype(int)

    # Train logit model
    logit_model = train_logit(X_train, y_train)
    
    # Evaluate the model 
    evaluate_model(logit_model, X_test, y_test, X_train, y_train)

if __name__ == '__main__': 
    main('../bert/embeddings.pickle')
    main('../bert/embeddings_without_emojis.pickle')