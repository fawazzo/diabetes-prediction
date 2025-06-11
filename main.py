# main.py
from utils.data_processing import load_and_preprocess_data
from models.decision_tree_model import train_decision_tree
from models.svm_model import train_svm
from models.knn_model import train_knn
from models.ann_model import train_ann
from sklearn.metrics import accuracy_score
import pickle
import tensorflow as tf

if __name__ == "__main__":
    # 0. Set up directories
    DATA_DIR = "data/"
    MODELS_DIR = "models/"

    # 1. Load and preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, scaler  = load_and_preprocess_data(DATA_DIR)

    with open(f'{MODELS_DIR}scaler.pkl', 'wb') as file:
        pickle.dump(scaler,file)

    # 2. Train models
    dt_model = train_decision_tree(X_train, y_train, max_depth=5)
    svm_model = train_svm(X_train, y_train, C=1.0)
    knn_model = train_knn(X_train, y_train, n_neighbors=5)
    ann_model = train_ann(X_train, y_train, epochs = 200)

    # 3. Evaluate models
    models = {
        "Decision Tree": (dt_model, f'{MODELS_DIR}decision_tree_model.pkl'),
        "SVM": (svm_model, f'{MODELS_DIR}svm_model.pkl'),
        "k-NN": (knn_model, f'{MODELS_DIR}knn_model.pkl'),
        "ANN": (ann_model, f'{MODELS_DIR}ann_model.h5')  # ANN model path
    }

    results = {}  # Store the results in a dic

    for model_name, (model, model_path) in models.items():
        if model_name == 'ANN':
            model = tf.keras.models.load_model(model_path)  # Load ANN model
            y_pred = (model.predict(X_val) > 0.5).astype("int32")
        else:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        results[model_name] = accuracy  # Store accuracy in the dictionary
        print(f"{model_name} Validation Accuracy: {accuracy}")

    best_model = max(results, key=results.get)  # Get best model based on validation accuracy
    print(f"\nBest performing model: {best_model} with accuracy: {results[best_model]}")

    # Evaluate and save the best model on the test set
    best_model_path = models[best_model][1]

    if best_model == 'ANN':
        best_model = tf.keras.models.load_model(best_model_path)
        y_test_pred = (best_model.predict(X_test) > 0.5).astype("int32")
    else:
        with open(best_model_path, 'rb') as file:
            best_model = pickle.load(file)
        y_test_pred = best_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"{best_model} Test Accuracy: {test_accuracy}")

    # ... (previous code from main.py)

    # 4. GUI (using Tkinter)
    import tkinter as tk
    from tkinter import ttk
    import numpy as np


    def predict_diabetes():
        try:
            pregnancies = float(pregnancies_entry.get())
            glucose = float(glucose_entry.get())
            blood_pressure = float(blood_pressure_entry.get())
            skin_thickness = float(skin_thickness_entry.get())
            insulin = float(insulin_entry.get())
            bmi = float(bmi_entry.get())
            dpf = float(dpf_entry.get())
            age = float(age_entry.get())

            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            input_data_scaled = scaler.transform(input_data)  # Important to scale the user input

            selected_model = model_var.get()
            model, model_path = models[selected_model]
            if selected_model == 'ANN':
                model = tf.keras.models.load_model(model_path)
                prediction = (model.predict(input_data_scaled) > 0.5).astype("int32")[0][0]
            else:
                with open(model_path, 'rb') as file:
                    model = pickle.load(file)
                prediction = model.predict(input_data_scaled)[0]

            if prediction == 1:
                result_label.config(text="Prediction: High Risk of Diabetes")
            else:
                result_label.config(text="Prediction: Low Risk of Diabetes")

        except ValueError:
            result_label.config(text="Invalid input. Please enter numeric values.")


    root = tk.Tk()
    root.title("Diabetes Prediction Expert System")

    # Model Selection
    model_var = tk.StringVar(value="Decision Tree")  # Default selected model
    model_label = ttk.Label(root, text="Select Model:")
    model_label.grid(row=0, column=0, padx=5, pady=5)
    model_dropdown = ttk.Combobox(root, textvariable=model_var, values=list(models.keys()))
    model_dropdown.grid(row=0, column=1, padx=5, pady=5)

    # Input Fields
    input_labels = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI",
                    "Diabetes Pedigree Function", "Age"]
    entry_variables = []
    for i, label_text in enumerate(input_labels):
        label = ttk.Label(root, text=label_text + ":")
        label.grid(row=i + 1, column=0, padx=5, pady=5, sticky=tk.W)
        entry = ttk.Entry(root)
        entry.grid(row=i + 1, column=1, padx=5, pady=5)
        entry_variables.append(entry)

    pregnancies_entry, glucose_entry, blood_pressure_entry, skin_thickness_entry, insulin_entry, bmi_entry, dpf_entry, age_entry = entry_variables

    # Predict Button
    predict_button = ttk.Button(root, text="Predict", command=predict_diabetes)
    predict_button.grid(row=len(input_labels) + 2, column=0, columnspan=2, pady=10)

    # Result Label
    result_label = ttk.Label(root, text="")
    result_label.grid(row=len(input_labels) + 3, column=0, columnspan=2, pady=10)

    root.mainloop()