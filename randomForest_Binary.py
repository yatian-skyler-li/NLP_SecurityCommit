from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import csv
import chardet

with open('negative_commits.csv', 'rb') as f:
    result = chardet.detect(f.read())

with open('negative_commits.csv', newline='', encoding=result['encoding']) as f:
    reader = csv.DictReader(f)
    data1 = [{'message': row['message'], 'category': 0} for i, row in enumerate(reader) if i < 7000]

with open('security_patches.csv', newline='', encoding= 'utf-8') as f:
    reader = csv.DictReader(f)
    data2 = [{'message': row['message'], 'category': 1} for i, row in enumerate(reader) if i < 7000]

combined_data = data1 + data2

X = [row['message'] for row in combined_data]
y = [row['category'] for row in combined_data]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = RandomForestClassifier()
clf.fit(X_train_vec, y_train)

accuracy = clf.score(X_test_vec, y_test)
print("Accuracy:", accuracy)