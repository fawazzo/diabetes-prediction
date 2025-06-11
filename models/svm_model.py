# models/svm_model.py
from sklearn.svm import SVC
import pickle

def train_svm(X_train, y_train, save_path='models/svm_model.pkl', save_scaler_path = 'models/svm_scaler.pkl', **kwargs):
    """Trains an SVM classifier and saves the model."""
    model = SVC(**kwargs)
    model.fit(X_train, y_train)
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)

    print("SVM model trained and saved.")
    return model