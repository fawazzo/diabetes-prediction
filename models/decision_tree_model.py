# models/decision_tree_model.py
from sklearn.tree import DecisionTreeClassifier
import pickle

def train_decision_tree(X_train, y_train, save_path='models/decision_tree_model.pkl', save_scaler_path = 'models/decision_tree_scaler.pkl', **kwargs):
    """Trains a Decision Tree classifier and saves the model."""
    model = DecisionTreeClassifier(**kwargs)
    model.fit(X_train, y_train)
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)
    print("Decision Tree model trained and saved.")
    return model