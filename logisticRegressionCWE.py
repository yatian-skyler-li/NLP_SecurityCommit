"""
Author: Yatian Li
Date: 30 May 2023
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Download nltk stop words, punkt and wordnet corpus
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Read the CSV file 'security_patches.csv'
df = pd.read_csv('security_patches.csv')

# Select only the rows where 'cwe_id' count is more than 400
frequent_values = df['cwe_id'].value_counts()
df = df[df['cwe_id'].isin(frequent_values[frequent_values > 400].index)]

# Remove any rows where 'message' or 'cwe_id' is null
df = df.dropna(subset=['message', 'cwe_id'])

# Initialize a set of English stop words and a WordNet lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define a function to preprocess the text by tokenizing, lemmatizing and removing stop words
def preprocess_text(text):
    word_tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in word_tokens if word not in stop_words]
    return " ".join(lemmatized)

# Apply the preprocessing function to the 'message' column
df['message'] = df['message'].apply(lambda x: preprocess_text(x))

# Split the dataframe into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['cwe_id'], test_size=0.2)

# Create a pipeline of TfidfVectorizer and LogisticRegression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear'))
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test data
y_pred = pipeline.predict(X_test)

# Create a dataframe of actual and predicted values
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Filter the dataframe to include only rows where 'Actual' is in 'frequent_cwe_ids'
frequent_cwe_ids = frequent_values[frequent_values > 400].index
filtered_results_df = results_df[results_df['Actual'].isin(frequent_cwe_ids)]

# Compute and print the accuracy for each CWE ID
accuracy_scores = {}
for cwe_id in frequent_cwe_ids:
    cwe_df = filtered_results_df[filtered_results_df['Actual'] == cwe_id]
    accuracy = accuracy_score(cwe_df['Actual'], cwe_df['Predicted'])
    accuracy_scores[cwe_id] = accuracy

print("Accuracy for each CWE IDs with frequency > 400:")
for cwe_id, accuracy in accuracy_scores.items():
    print(f"CWE ID {cwe_id}: {accuracy}")

# Compute and print the general accuracy, f1 score and precision for all CWE IDs with frequency > 400
# set zero_divsion = 1 since some cweid may not exist in test data if we set the frequency threshold too low
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=1))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=1))