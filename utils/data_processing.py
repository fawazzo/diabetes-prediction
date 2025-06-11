# utils/data_processing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_dir="data/"):
    """Loads data from CSV files, handles missing values, and scales features."""

    X_train = pd.read_csv(f'{data_dir}train_data.csv')
    y_train = pd.read_csv(f'{data_dir}train_labels.csv').squeeze()
    X_val = pd.read_csv(f'{data_dir}val_data.csv')
    y_val = pd.read_csv(f'{data_dir}val_labels.csv').squeeze()
    X_test = pd.read_csv(f'{data_dir}test_data.csv')
    y_test = pd.read_csv(f'{data_dir}test_labels.csv').squeeze()

    # Handle missing values (example: impute with mean)
    X_train = X_train.fillna(X_train.mean())
    X_val = X_val.fillna(X_val.mean())
    X_test = X_test.fillna(X_test.mean())

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler