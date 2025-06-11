# models/knn_model.py
from sklearn.neighbors import KNeighborsClassifier
import pickle

def train_knn(X_train, y_train, save_path='models/knn_model.pkl', save_scaler_path = 'models/knn_scaler.pkl', **kwargs):
    """Trains a k-NN classifier and saves the model."""
    model = KNeighborsClassifier(**kwargs)
    model.fit(X_train, y_train)
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)

    print("k-NN model trained and saved.")
    return model