"""
Author: Yatian Li
Date: 30 May 2023
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

# Downloading necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Reading CSV data into pandas dataframe
df = pd.read_csv('security_patches.csv')

# Counting the frequency of each 'cwe_id' and selecting those with count > 400
frequent_values = df['cwe_id'].value_counts()
frequent_cwe_ids = frequent_values[frequent_values > 400].index
df = df[df['cwe_id'].isin(frequent_values[frequent_values > 400].index)]

# Dropping any rows with missing 'message' or 'cwe_id'
df = df.dropna(subset=['message', 'cwe_id'])

# Initializing NLTK's English stopwords and WordNet lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Defining a text preprocessing function
def preprocess_text(text):
    word_tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(word) for word in word_tokens if word not in stop_words]
    return " ".join(lemmatized)

# Applying the text preprocessing function to the 'message' column
df['message'] = df['message'].apply(lambda x: preprocess_text(x))

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['cwe_id'], test_size=0.2)

# Creating a pipeline for transforming the data and fitting the model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Transform the text data into TF-IDF matrix
    ('clf', SVC(kernel='linear'))  # Use a linear SVC model for classification
])

# Training the model
pipeline.fit(X_train, y_train)

# Making predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate and print the accuracy for each frequent CWE ID
accuracy_scores = {}
for cwe_id in frequent_cwe_ids:
    mask = y_test == cwe_id  # Get a boolean mask for the samples of the current CWE ID
    accuracy = accuracy_score(y_test[mask], y_pred[mask])  # Calculate the accuracy for the current CWE ID
    accuracy_scores[cwe_id] = accuracy
    print(f"CWE ID {cwe_id}: Accuracy {accuracy}")

# Print the overall accuracy, f1 score, and precision of the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted', zero_division=1))
print("Precision:", precision_score(y_test, y_pred, average='weighted', zero_division=1))
