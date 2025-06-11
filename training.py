import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(csv_file, target_column='Outcome', train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Splits a CSV file into training, validation, and test sets and saves them to separate files.

    Args:
      csv_file (str): Path to the CSV file.
      target_column (str): Name of the target variable column. Default is 'Outcome'.
      train_ratio (float): Ratio of data for training (0.0 to 1.0). Default is 0.7.
      val_ratio (float): Ratio of data for validation (0.0 to 1.0). Default is 0.15.
      test_ratio (float): Ratio of data for testing (0.0 to 1.0). Default is 0.15.
      random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
      None (saves the files directly)
    """

    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    # 1. Load the data
    df = pd.read_csv(csv_file)

    # 2. Separate features (X) and target (y)
    X = df.drop(columns=target_column)
    y = df[target_column]

    # 3. Split into training and the rest (temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=random_state, stratify=y
    )

    # 4. Split the rest into validation and test
    if test_ratio > 0:
      X_val, X_test, y_val, y_test = train_test_split(
          X_temp, y_temp, test_size=(test_ratio / (val_ratio + test_ratio)),
          random_state=random_state, stratify=y_temp
      )
    else:
      X_val, y_val = X_temp, y_temp
      X_test, y_test = pd.DataFrame(), pd.Series() # Empty dataframes for test data

    # 5. Save the datasets to CSV files
    X_train.to_csv('train_data.csv', index=False)
    y_train.to_csv('train_labels.csv', index=False, header=True)
    X_val.to_csv('val_data.csv', index=False)
    y_val.to_csv('val_labels.csv', index=False, header=True)
    X_test.to_csv('test_data.csv', index=False)
    y_test.to_csv('test_labels.csv', index=False, header=True)

    print("Data split and saved to files: train_data.csv, val_data.csv, test_data.csv")
    print("Labels split and saved to files: train_labels.csv, val_labels.csv, test_labels.csv")

# Example Usage:
csv_file = 'data/diabetes knn.csv'  # Replace with your CSV file path
split_data(csv_file)