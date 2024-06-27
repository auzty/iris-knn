# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd

# Load the Iris dataset
df = pd.read_csv('datasets/iris.data.csv')
#iris = load_iris()
X = df.drop('target', axis=1)#iris.data
y = df['target'] #iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (optional but recommended for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Train the classifier
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Sample Iris-setosa data
original_data_point = [4.9,3.0,1.4,0.2] 

# Transform the original data point using the fitted scaler
original_data_point_scaled = scaler.transform([original_data_point])

# Predict the class for the sample data point
sample_prediction = knn.predict(original_data_point_scaled)

# Print the Prediction result
print(sample_prediction)