"""
Yatian Li
30 May 2023
"""
import torch
import csv
import chardet
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Detecting the encoding of 'negative_commits.csv' using chardet
with open('negative_commits.csv', 'rb') as f:
    result = chardet.detect(f.read())

# Reading 'negative_commits.csv' with the detected encoding
with open('negative_commits.csv', newline='', encoding=result['encoding']) as f:
    # Creating a list of dictionaries for the commit messages with category 0 (negative commits)
    reader = csv.DictReader(f)
    data1 = [{'message': row['message'], 'category': 0} for i, row in enumerate(reader) if i < 7000]

# Reading 'security_patches.csv' assuming it is encoded in UTF-8
with open('security_patches.csv', newline='', encoding= 'utf-8') as f:
    # Creating a list of dictionaries for the commit messages with category 1 (security patches)
    reader = csv.DictReader(f)
    data2 = [{'message': row['message'], 'category': 1} for i, row in enumerate(reader) if i < 7000]

# Combining the data from 'negative_commits.csv' and 'security_patches.csv'
combined_data = data1 + data2

# Initializing a BERT tokenizer using the 'bert-base-uncased' pre-trained model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Extracting the commit messages and labels from the combined data
texts = [item['message'] for item in combined_data]
labels = [item['category'] for item in combined_data]

# Encoding the commit messages using the tokenizer, applying truncation, padding, and limiting the maximum length
encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=128)

# Splitting the texts and labels into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# Custom dataset class for the Git dataset
class GitDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Loading the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Training arguments for the Trainer object
training_args = TrainingArguments(
    output_dir='./results',  # Output directory for model predictions and checkpoints
    num_train_epochs=10,     # Total number of training epochs
    per_device_train_batch_size=16,  # Batch size per device during training
    per_device_eval_batch_size=64,   # Batch size for evaluation
    warmup_steps=500,      # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,     # Strength of weight decay
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=10,      # Log & save weights each logging_steps
    evaluation_strategy='steps',     # Evaluation strategy to adopt during training
    eval_steps=500,        # Evaluation step
    save_steps=500,        # After # steps model is saved
    load_best_model_at_end=True,     # Load the best model when finished training (default metric is loss)
    metric_for_best_model='accuracy', # Use accuracy to determine the best model
)

# Encoding the training and validation texts using the tokenizer
train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=128)

# Creating instances of the GitDataset for the training and validation datasets
train_dataset = GitDataset(train_encodings, train_labels)
val_dataset = GitDataset(val_encodings, val_labels)

# Function to compute evaluation metrics
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

# Creating a Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Training the model
trainer.train()

# Evaluating the trained model on the validation dataset
eval_results = trainer.evaluate()

# Printing the accuracy of the evaluation results
print("Accuracy: ", eval_results['eval_accuracy'])