import pandas as pd
import numpy as np
import csv
import chardet
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

with open('negative_commits.csv', 'rb') as f:
    result = chardet.detect(f.read())

with open('negative_commits.csv', newline='', encoding=result['encoding']) as f:
    reader = csv.DictReader(f)
    data1 = [{'message': row['message'], 'category': 0} for i, row in enumerate(reader) if i < 8000]

with open('security_patches.csv', newline='', encoding= 'utf-8') as f:
    reader = csv.DictReader(f)
    data2 = [{'message': row['message'], 'category': 1} for i, row in enumerate(reader) if i < 8000]

combined_data = data1 + data2
combined_data = pd.DataFrame(combined_data)
combined_data['category'] = combined_data['category'].astype(int)

train_data, test_data = train_test_split(combined_data, test_size=0.2)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data['message'])

X_train = tokenizer.texts_to_sequences(train_data['message'])
X_test = tokenizer.texts_to_sequences(test_data['message'])

# Pad sequences
max_len = 100
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Define
model = Sequential()
model.add(Embedding(5000, 128, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X_train, train_data['category'], batch_size=32, epochs=10)

# Evaluate
predictions = model.predict(X_test)
predictions = np.round(predictions).astype(int)
scores = model.evaluate(X_test, test_data['category'], verbose=0)
precision = precision_score(test_data['category'], predictions)
f1 = f1_score(test_data['category'], predictions)
print('Accuracy:', scores[1], "Precision: ", precision, "f1: ", f1)

