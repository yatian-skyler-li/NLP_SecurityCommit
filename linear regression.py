"""
Yatian Li
30 May 2023
This code is for training and evaluating a Logistic Regression model on two types of commit messages.
The messages are vectorized using a CountVectorizer and then passed to the model for training. 
After training, the model is evaluated on a test set and the accuracy, precision, and f1-score are calculated.
"""
import pandas as pd
import csv
import chardet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score

# Reading the 'negative_commits.csv' file and detecting its encoding
with open('negative_commits.csv', 'rb') as f:
    result = chardet.detect(f.read())

# Loading the 'negative_commits.csv' file with detected encoding and creating a list of dictionaries from it
with open('negative_commits.csv', newline='', encoding=result['encoding']) as f:
    reader = csv.DictReader(f)
    data1 = [{'message': row['message'], 'category': 0} for i, row in enumerate(reader) if i < 8000]

# Loading the 'security_patches.csv' file and creating a list of dictionaries from it
with open('security_patches.csv', newline='', encoding= 'utf-8') as f:
    reader = csv.DictReader(f)
    data2 = [{'message': row['message'], 'category': 1} for i, row in enumerate(reader) if i < 8000]

# Combining the two lists of dictionaries into a single list and converting it to a DataFrame
combined_data = data1 + data2
combined_data = pd.DataFrame(combined_data)

# Filling NaN values in the DataFrame with an empty string
combined_data = combined_data.fillna('')

# Initializing a CountVectorizer with English stop words and n-gram range of 1-2
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))

# Vectorizing the 'message' column in the DataFrame
X = vectorizer.fit_transform(combined_data['message'])

# Getting the 'category' column as labels
y = combined_data['category']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initializing a Logistic Regression model
lr_model = LogisticRegression()

# Training the model with the training data
lr_model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = lr_model.predict(X_test)

# Calculating precision, f1-score, and accuracy of the model's predictions
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Printing the calculated metrics
print(f"Accuracy: {accuracy}, Precision: {precision}, F1-score: {f1}")