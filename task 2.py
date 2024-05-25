import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the datasets
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# Check the columns
print("Train data columns:", train_data.columns)
print("Test data columns:", test_data.columns)



# Separate features and target variable
X_train = train_data.drop('target', axis=1)
y_train = train_data['target']
X_test = test_data



#preprocessing the data
# Standardize the datasets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#train classification model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize models
models = {
   # "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

#logistic regression
from sklearn.linear_model import LogisticRegression

# Initialize Logistic Regression model with a higher number of iterations
logistic_model = LogisticRegression(max_iter=1000)

# Train Logistic Regression model
logistic_model.fit(X_train_scaled, y_train)

# Check accuracy
train_accuracy = accuracy_score(y_train, logistic_model.predict(X_train_scaled))

# Predict target values for the test set
test_predictions_logistic = logistic_model.predict(X_test_scaled)

# 


# Train models and record their performance
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
    test_predictions = model.predict(X_test_scaled)
    results[name] = {
        "train_accuracy": train_accuracy,
        "test_predictions": test_predictions
    }

results
#Record results
results["Logistic Regression"] = {
    "train_accuracy": train_accuracy,
    "test_predictions": test_predictions_logistic.tolist()
}

# # Prepare results for output
# results_output = {}
# for name, result in results.items():
#     results_output[name] = {
#         "train_accuracy": result["train_accuracy"],
#         "test_predictions": result["test_predictions"].tolist()  # Convert to list for JSON compatibility
#     }

# results_output
# Print target values for each model
for name, result in results.items():
    print(f"Test predictions for {name}: {result['test_predictions']}")


