# models/ann_model.py
from tensorflow import keras
import pickle

def train_ann(X_train, y_train, save_path='models/ann_model.h5', input_shape = 8 ,save_scaler_path = 'models/ann_scaler.pkl', **kwargs):
  """Trains an Artificial Neural Network and saves the model."""
  model = keras.Sequential([
      keras.layers.Dense(12, activation='relu', input_shape=(input_shape,)),
      keras.layers.Dense(8, activation='relu'),
      keras.layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

  model.fit(X_train, y_train, **kwargs)
  model.save(save_path)
  print("ANN model trained and saved.")
  return model