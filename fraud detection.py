import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Load dataset (assuming the dataset is in the same directory)
# The dataset should be in a CSV file format
data = pd.read_csv('credit_card_transactions.csv')

# Display the first few rows of the dataset to understand its structure
print(data.head())

# Preprocess the data
# Remove the 'Time' column as it is not useful for fraud detection
data = data.drop(['Time'], axis=1)

# Separate features (X) and target variable (y)
X = data.drop(['Class'], axis=1)  # Features
y = data['Class']  # Target variable

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Anomaly Detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
iso_forest.fit(X_train)  # Train the model on the training data

# Predict anomalies in the test set
y_pred_iso = iso_forest.predict(X_test)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)  # Convert -1 to 1 for fraud, 1 to 0 for non-fraud

# Print evaluation metrics for Isolation Forest
print("Isolation Forest Classification Report:")
print(classification_report(y_test, y_pred_iso, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_iso))
print("Accuracy:", accuracy_score(y_test, y_pred_iso))

# Classification using Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)  # Train the model on the training data

# Predict on the test set
y_pred_rf = rf_clf.predict(X_test)

# Print evaluation metrics for Random Forest
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
