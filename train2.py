# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, MT5ForConditionalGeneration, MT5TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from datasets import load_dataset, load_metric
import numpy as np
import torch

torch.cuda.empty_cache()
torch.manual_seed(0)
NUM_EPOCHS = 3 if torch.cuda.is_available() else 1
PERCENTILES = (80, 100)

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
LEARNING_RATE = 5e-05



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
#from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback


# Read data

def read_data(path):
    data = {'tokens': [], 'tags': []}
    with open(path, 'r', encoding='utf-8') as f:
        f=f.readlines()
        for line in f:
            l = line.strip()
            data['tags'].append(l)
            texts = []
            for pair in l.split():
                word, _ = pair.split('_')
                texts.append(word)
            data['tokens'].append(''.join(texts))

    return data


def get_max_length(data):
    lengths = [len(i) for i in data]
    return int(np.percentile(lengths, 80)) +1


data = read_data('udp/train.txt')
tokenizer = MT5TokenizerFast.from_pretrained('mt5tokenizer')
model = MT5ForConditionalGeneration.from_pretrained('mt5small')


X = data["tokens"]
y = data["tags"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from IPython import embed; embed()
X_train_tokenized = tokenizer.encode_plus(X_train, padding=True, truncation=True, max_length=get_max_length(X_train))
#TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
X_val_tokenized = tokenizer.encode_plus(X_val, padding=True, truncation=True, max_length=get_max_length(X_val))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


args = Seq2SeqTrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    weight_decay=0.01,
    logging_dir='./logs/',
    logging_steps=100,
    learning_rate=5e-05,
    warmup_steps=200,
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_steps=3000,
    seed=0,
    load_best_model_at_end=True,
    predict_with_generate=True,
)
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train pre-trained model
trainer.train()

# Load test data
test_data = read_data("udp/test.txt")
X_test = test_data["review"]
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=get_max_length(X_test))

# Create torch dataset
test_dataset = Dataset(X_test_tokenized)

# Load trained model
model_path = "output/checkpoint-50000"
model = MT5ForConditionalGeneration.from_pretrained(model_path, num_labels=2)

# Define test trainer
test_trainer = Trainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)