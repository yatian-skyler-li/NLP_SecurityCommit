import torch
import csv
import chardet
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

with open('negative_commits.csv', 'rb') as f:
    result = chardet.detect(f.read())
with open('negative_commits.csv', newline='', encoding=result['encoding']) as f:
    reader = csv.DictReader(f)
    data1 = [{'message': row['message'], 'category': 0} for i, row in enumerate(reader) if i < 7000]
with open('security_patches.csv', newline='', encoding= 'utf-8') as f:
    reader = csv.DictReader(f)
    data2 = [{'message': row['message'], 'category': 1} for i, row in enumerate(reader) if i < 7000]

combined_data = data1 + data2
tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')

texts = [item['message'] for item in combined_data]
labels = [item['category'] for item in combined_data]
encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=128)

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

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

model = AutoModelForSequenceClassification.from_pretrained('microsoft/graphcodebert-base', num_labels=2)

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

train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=128)

train_dataset = GitDataset(train_encodings, train_labels)
val_dataset = GitDataset(val_encodings, val_labels)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
eval_results = trainer.evaluate()
print("Accuracy: ", eval_results['eval_accuracy'])
