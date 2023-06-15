"""
Yatian Li
30 May 2023
"""
import torch
import csv
import chardet
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load negative commit messages and assign label 0
with open('negative_commits.csv', 'rb') as f:
    result = chardet.detect(f.read())
with open('negative_commits.csv', newline='', encoding=result['encoding']) as f:
    reader = csv.DictReader(f)
    data1 = [{'message': row['message'], 'category': 0} for i, row in enumerate(reader) if i < 7000]

# Load security patch messages and assign label 1
with open('security_patches.csv', newline='', encoding= 'utf-8') as f:
    reader = csv.DictReader(f)
    data2 = [{'message': row['message'], 'category': 1} for i, row in enumerate(reader) if i < 7000]

# Combine data from both negative commits and security patches
combined_data = data1 + data2

# Specify the pretrained model
model_name = 'microsoft/codebert-base'

# Load the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Extract texts and labels from the data
texts = [item['message'] for item in combined_data]
labels = [item['category'] for item in combined_data]

# Tokenize the texts
encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=128)

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# Define a PyTorch Dataset class
class GitDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # Return an item from the dataset
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    # Return the length of the dataset
    def __len__(self):
        return len(self.labels)

# Load the pretrained model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Define training arguments
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
)

# Tokenize the training and validation texts
train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=128)

# Create training and validation datasets
train_dataset = GitDataset(train_encodings, train_labels)
val_dataset = GitDataset(val_encodings, val_labels)

# Import metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define a function to compute metrics
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

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print the accuracy
print("Accuracy: ", eval_results['eval_accuracy'])
