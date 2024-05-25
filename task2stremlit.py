import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title of the Streamlit app
st.title("Machine Learning Model Comparison")

# Upload train and test datasets
train_file = st.file_uploader("Upload Train Dataset", type=["xlsx"])
test_file = st.file_uploader("Upload Test Dataset", type=["xlsx"])

if train_file and test_file:
    # Load the datasets
    train_data = pd.read_excel(train_file)
    test_data = pd.read_excel(test_file)

    # Display the columns
    st.write("Train data columns:", train_data.columns)
    st.write("Test data columns:", test_data.columns)

    # Separate features and target variable
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_test = test_data

    # Preprocessing the data
    # Standardize the datasets
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}

    # Train models and record their performance
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
        test_predictions = model.predict(X_test_scaled)
        results[name] = {
            "train_accuracy": train_accuracy,
            "test_predictions": test_predictions.tolist()
        }

    # Display results
    for name, result in results.items():
        st.write(f"Model: {name}")
        st.write(f"Train Accuracy: {result['train_accuracy']}")
        st.write(f"Test Predictions: {result['test_predictions']}")
else:
    st.write("Please upload both train and test datasets.")
