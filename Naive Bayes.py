"""
Author: Yatian Li
Date: 30 May 2023
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score

# Load the CSV file into a DataFrame
df = pd.read_csv('security_patches.csv')
frequent_values = df['cwe_id'].value_counts()
df = df[df['cwe_id'].isin(frequent_values[frequent_values > 400].index)]

# Drop rows with NaN in 'message' column
df = df.dropna(subset=['message', 'cwe_id'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['cwe_id'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer and transform the messages
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Multinomial Naive Bayes classifier
clf = MultinomialNB().fit(X_train_tfidf, y_train)

# Predict the CWE IDs for the test set
y_pred = clf.predict(X_test_tfidf)

# Calculate and print the accuracy for each CWE ID
accuracy_scores = {}
for cwe_id in frequent_values[frequent_values > 400].index:
    mask = y_test == cwe_id
    accuracy = accuracy_score(y_test[mask], y_pred[mask])
    accuracy_scores[cwe_id] = accuracy
    print(f"CWE ID {cwe_id}: Accuracy {accuracy}")

# Print overall evaluation metrics
print("Overall Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=1))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=1))