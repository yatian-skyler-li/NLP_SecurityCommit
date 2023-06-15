"""
Compare with original codeBERT.py, there are some potential improve. Due to the time limitation,
I haven't test the accuracy with different training arguments

-Shuffles the combined data before splitting.
-Moves the tokenization step right after the split.
-Introduces a placeholder for text preprocessing.
-Uses the f1 score to determine the best model.
-Saves the model after training.
"""

import torch
import csv
import chardet
import random
import string
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    return text

# Load and preprocess data from negative_commits.csv
with open('negative_commits.csv', 'rb') as f:
    result = chardet.detect(f.read())
with open('negative_commits.csv', newline='', encoding=result['encoding']) as f:
    reader = csv.DictReader(f)
    data1 = [{'message': preprocess_text(row['message']), 'category': 0} for i, row in enumerate(reader) if i < 7000]

# Load and preprocess data from security_patches.csv
with open('security_patches.csv', newline='', encoding= 'utf-8') as f:
    reader = csv.DictReader(f)
    data2 = [{'message': preprocess_text(row['message']), 'category': 1} for i, row in enumerate(reader) if i < 7000]

# Combine the two datasets
combined_data = data1 + data2
random.shuffle(combined_data)  # Shuffling the combined data

# Define the model and tokenizer
model_name = 'microsoft/codebert-base' 
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Extract texts and labels from combined data
texts = [item['message'] for item in combined_data]
labels = [item['category'] for item in combined_data]

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# Tokenize the training and validation texts
train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=128)

# Define a dataset class
class GitDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # Fetch an item from the dataset
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    # Get the length of the dataset
    def __len__(self):
        return len(self.labels)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=2,
    warmup_steps=50,
    weight_decay=0.1,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy='steps', 
    evaluation_strategy='steps',  
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

# Prepare the training and validation datasets
train_dataset = GitDataset(train_encodings, train_labels)
val_dataset = GitDataset(val_encodings, val_labels)

# Import metrics for evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Create a trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model
trainer.save_model()

# Evaluate the model and print results
eval_results = trainer.evaluate()
print("Accuracy: ", eval_results['eval_accuracy'])
print("F1 Score: ", eval_results['eval_f1'])
