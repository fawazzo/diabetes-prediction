# 🩺 Diabetes Prediction Expert System

![Python Logo](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn Logo](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow Logo](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit Logo](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Tkinter Logo](https://img.shields.io/badge/Tkinter-C41221?style=for-the-badge&logo=python&logoColor=white) 

---

## 🎯 Project Overview

This project implements a robust Python-based expert system for predicting the risk of diabetes. It leverages various machine learning algorithms, comprehensive data preprocessing, and offers **two distinct user interfaces** – a standalone Tkinter GUI and a modern Streamlit web application – for interactive predictions.

The system is designed to preprocess patient health data, train and evaluate multiple classification models (including Decision Trees, Support Vector Machines, k-Nearest Neighbors, and Artificial Neural Networks), and then use the best-performing model to provide an instant diabetes risk assessment.

**Key Features:**

*   **Comprehensive Data Preprocessing:** Handles data loading, splitting into train/validation/test sets, and scaling of features.
*   **Multiple Machine Learning Models:** Implements and compares diverse classification algorithms.
*   **Model Persistence:** Trained models and the data scaler are saved for reusability without re-training.
*   **Extensive Model Evaluation:** Calculates and displays validation and test accuracy for all models to identify the best performer.
*   **Interactive Tkinter GUI:** A desktop application allowing users to input parameters and get a prediction locally.
*   **Modern Streamlit Web App:** A user-friendly web interface for predictions, with dynamic risk percentage and visual indicators.
*   **Scalability & Modularity:** Organized code into clear modules (`utils`, `models`) for better maintainability and understanding.

---

## ✨ Technologies Used

*   **Python:** The core programming language.
*   **Pandas:** For efficient data manipulation and analysis.
*   **NumPy:** Essential for numerical operations.
*   **Scikit-learn:** Comprehensive library for classical machine learning algorithms, model training, and evaluation.
*   **TensorFlow/Keras:** For building and training the Artificial Neural Network (ANN) model.
*   **Tkinter:** Python's standard GUI toolkit for the desktop application.
*   **Streamlit:** For creating the interactive web-based prediction interface.

---

## 📂 Project Structure

Diabetes-Prediction-Expert-System/
├── data/
│ └── diabetes knn.csv # The primary dataset (e.g., Pima Indians Diabetes Dataset)
│ ├── train_data.csv # Processed training features
│ ├── train_labels.csv # Processed training labels
│ ├── val_data.csv # Processed validation features
│ ├── val_labels.csv # Processed validation labels
│ ├── test_data.csv # Processed test features
│ └── test_labels.csv # Processed test labels
├── models/
│ ├── scaler.pkl # Trained StandardScaler for feature scaling
│ ├── decision_tree_model.pkl # Saved Decision Tree model
│ ├── svm_model.pkl # Saved SVM model
│ ├── knn_model.pkl # Saved k-NN model
│ └── ann_model.h5 # Saved Artificial Neural Network model
├── src/
│ ├── utils/
│ │ └── data_processing.py # Script for loading, preprocessing, and splitting data
│ └── models/
│ ├── decision_tree_model.py # Defines and trains Decision Tree model
│ ├── svm_model.py # Defines and trains SVM model
│ ├── knn_model.py # Defines and trains k-NN model
│ └── ann_model.py # Defines and trains Artificial Neural Network model
├── main.py # Orchestrates data loading, model training, evaluation, and runs Tkinter GUI
├── streamlit_app.py # The Streamlit web application script
├── requirements.txt # List of Python dependencies
├── README.md # This file
└── .gitignore # Specifies intentionally untracked files to ignore

---

## 🚀 How to Run the Project

Follow these steps to set up and run the Diabetes Prediction Expert System locally.

### Prerequisites

*   Python 3.8+ installed on your system.

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/FawazKourd/Diabetes-Prediction-Expert-System.git
    cd Diabetes-Prediction-Expert-System
    ```
    *(Replace `FawazKourd` with your actual GitHub username)*

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(A `requirements.txt` file is essential. See "Creating `requirements.txt`" below.)*

### Step-by-Step Usage

1.  **Prepare the Data and Train Models:**
    The `main.py` script handles data loading, preprocessing, model training, evaluation, and saves the trained models and scaler.
    ```bash
    python main.py
    ```
    This script will:
    *   Load `data/diabetes knn.csv`.
    *   Split the data into training, validation, and test sets, saving them to `data/`.
    *   Train Decision Tree, SVM, k-NN, and ANN models.
    *   Save all trained models and the `scaler.pkl` to the `models/` directory.
    *   Print validation and test accuracies for all models, identifying the best performer.
    *   **Automatically launch the Tkinter GUI** at the end.

2.  **Using the Tkinter Desktop GUI (Launches automatically after `main.py`):**
    *   After running `python main.py`, a desktop window will appear.
    *   Select your desired machine learning model from the dropdown.
    *   Enter the patient's health parameters into the respective fields.
    *   Click "Predict" to see the diabetes risk assessment.

3.  **Using the Streamlit Web Application:**
    Once models are trained (by running `main.py` at least once), you can launch the interactive web application:
    ```bash
    streamlit run streamlit_app.py
    ```
    *   This command will open the Streamlit app in your default web browser (usually `http://localhost:8501`).
    *   Select a model, input patient data, and observe the dynamic risk prediction.

---

## 📊 Dataset

This project primarily uses the **Pima Indians Diabetes Database**, found in `data/diabetes knn.csv`. This dataset is widely used for classification tasks and contains various diagnostic measurements and a binary outcome variable indicating diabetes onset.

**Key Features in the Dataset:**

*   `Pregnancies`: Number of times pregnant.
*   `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
*   `BloodPressure`: Diastolic blood pressure (mm Hg).
*   `SkinThickness`: Triceps skin fold thickness (mm).
*   `Insulin`: 2-Hour serum insulin (mu U/ml).
*   `BMI`: Body mass index (weight in kg/(height in m)^2).
*   `DiabetesPedigreeFunction`: Diabetes pedigree function.
*   `Age`: Age in years.
*   `Outcome`: Class variable (0 = non-diabetic, 1 = diabetic).

---

## 📈 Model Performance and Evaluation

The system thoroughly evaluates each trained model on a dedicated validation set and identifies the best performer. The final chosen model is then evaluated on an unseen test set to report its generalization accuracy.

**Evaluation Metrics Used:**
*   **Accuracy Score:** The proportion of correctly classified instances.

The project demonstrates a robust approach to model selection, ensuring that the prediction system relies on the most effective algorithm for the given dataset.

---

## 📝 Creating `requirements.txt`

To ensure all necessary Python packages are installed, create a `requirements.txt` file in your project's root directory. You can generate this automatically after installing all dependencies:

```bash
pip freeze > requirements.txt
🤝 Contribution
This project was developed by Fawaz KOURDOUGHLI.
Feel free to fork the repository, open issues, or submit pull requests for any improvements or bug fixes.
📄 License
This project is open-source and available under the MIT License.
(Add a LICENSE file in your repository if you want to make it open source)
📧 Contact
For any inquiries or feedback, please contact me at:
Fawaz KOURDOUGHLI
fawaz.kourdoughli@gmail.com
