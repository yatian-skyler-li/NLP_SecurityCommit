"""
Author: Yatian Li
Date: 30 May 2023
"""
import pandas as pd
import csv
import chardet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Detecting the encoding of the 'negative_commits.csv' file
with open('negative_commits.csv', 'rb') as f:
    result = chardet.detect(f.read())

# Reading and parsing 'negative_commits.csv' file into a list of dictionaries
with open('negative_commits.csv', newline='', encoding=result['encoding']) as f:
    reader = csv.DictReader(f)
    data1 = [{'message': row['message'], 'category': 0} for i, row in enumerate(reader)]

# Reading and parsing 'security_patches.csv' file into a list of dictionaries
with open('security_patches.csv', newline='', encoding= 'utf-8') as f:
    reader = csv.DictReader(f)
    data2 = [{'message': row['message'], 'category': 1} for i, row in enumerate(reader)]

# Combining the data from 'negative_commits.csv' and 'security_patches.csv' into a single dataframe
combined_data = data1 + data2
combined_data = pd.DataFrame(combined_data)

# Filling any NaN values in the dataframe with ''
combined_data = combined_data.fillna('')

# Creating a TF-IDF vectorizer and fitting it to the 'message' data
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(combined_data['message'])

# Transforming the 'message' data into a TF-IDF matrix
X = vectorizer.transform(combined_data['message'])

# Using the 'category' column as the target
y = combined_data['category']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Creating an SVM model with a linear kernel
svm_model = SVC(kernel='linear', C=1, gamma='auto')
svm_model.fit(X_train, y_train)

# Making predictions on the testing set
y_pred = svm_model.predict(X_test)

# Calculating the precision, f1 score, and accuracy of the model
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Printing the model's precision, f1 score, and accuracy
print(f"Accuracy: ", accuracy, "precision: ", precision, "f1: ", f1)
