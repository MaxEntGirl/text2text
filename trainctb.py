# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, MT5ForConditionalGeneration, MT5TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from datasets import load_dataset, load_metric
import numpy as np
import torch
from collections import Counter
torch.cuda.empty_cache()
import os
#os.environ['CUDA_VISIBLE_DEVICES']='1'
torch.manual_seed(0)
import sys
import csv
sys.path.append('/home/di/Desktop/thesis/')
NUM_EPOCHS = 20
PERCENTILES = (80, 100)

TRAIN_BATCH_SIZE = 8         # maybe more stable update with bigger batch, gradient_accumulation_steps trade off with batch size
EVAL_BATCH_SIZE = 1
WARMUP_STEPS = 200
#WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
LEARNING_RATE = 5e-05


train = []
with open('ctb/train.tsv') as f:   #共1342249行，55999句
    tsvreader = csv.reader(f, delimiter='\t',quoting=csv.QUOTE_NONE)
    for line in tsvreader:
        container = []
        if line != []:
            container.append(line)
        else:
            train.append(container)
            container = []
            continue


def get_max_length(tokenizer, train_dataset, column, percentile):
    def get_lengths(batch):
        return tokenizer(batch, padding=False, return_length=True)

    lengths = train_dataset.map(get_lengths, input_columns=column, batched=True)['length']
    print(int(np.percentile(lengths, percentile)) +1)
    return int(np.percentile(lengths, percentile)) +1


tokenizer = MT5TokenizerFast.from_pretrained('mt5tokenizer')
dataset = load_dataset('universal_dependencies','zh_gsdsimp')
#from IPython import embed; embed()
dataset = dataset.map(reformat_for_postag)
#dataset.save_to_disk()
print(dataset['train']['tgt_texts'][:10])

#model_name = "allenai/unifiedqa-t5-small" # you can specify the model size here
#tokenizer = AutoTokenizer.from_pretrained(model_name)


max_length = get_max_length(tokenizer, dataset['train'], 'text', PERCENTILES[0])
max_target_length = get_max_length(tokenizer, dataset['train'], 'tgt_texts', PERCENTILES[1])
# use unique chr/digits instead of tags

def tokenize(batch): 
    return tokenizer.prepare_seq2seq_batch(src_texts=batch['text'], 
                                         tgt_texts=batch['tgt_texts'], 
                                         max_length=max_length, 
                                         truncation=True,
                                         max_target_length=120,
                                         padding='max_length')

dataset = dataset.map(tokenize, batched=True)
print(dataset['train']['input_ids'][:3])
#dataset.set_format('torch', columns=['input_ids', 'labels', 'attention_mask'])

print(tokenizer.batch_decode(dataset['train']['labels'][:3][:10]))


#from IPython import embed; embed()

def ud_metrics(eval_prediction): # write a new one with F1 or sth else

    predictions = tokenizer.batch_decode(eval_prediction.predictions,
                                       skip_special_tokens=True)
    references = tokenizer.batch_decode(eval_prediction.label_ids,
                                   skip_special_tokens=True)
    truep,ref,pred = 0,0,0
    for p, r in zip(predictions,references):
        p = set(p.split('/'))
        r = set(r.split('/'))
        tp = len(p.intersection(r))
        truep+=tp
        ref+=len(r)
        pred+=len(p)

    precision = truep / pred
    recall = truep / ref
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}

'''    predictions = [{'id': str(i), 'prediction': pred.strip().lower()} \
                 for i, pred in enumerate(predictions)]
    references = [{'id': str(i), 'reference': ref.strip().lower()} \
                for i, ref in enumerate(references)]'''

model = MT5ForConditionalGeneration.from_pretrained('mt5small')
'''device = torch.device("cpu")
model.to(device)
print(next(model.parameters()).device)'''

training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
#    weight_decay=WEIGHT_DECAY,
    logging_dir='./logs/',
    evaluation_strategy="epoch",
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    predict_with_generate=True,
)

model.get_output_embeddings().weight.requires_grad=False
model.get_input_embeddings().weight.requires_grad=False

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
#    optimizers=(torch.optim.SGD(model.parameters(),lr=LEARNING_RATE),None),
    compute_metrics=ud_metrics,   #Must take a:class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
)

#print(trainer.evaluate(num_beams=2))
'''  File "/home/di/Desktop/thesis/train1.py", line 132, in <module>
    print(trainer.evaluate(num_beams=2))
  File "/home/di/anaconda3/lib/python3.7/site-packages/transformers/trainer_seq2seq.py", line 74, in evaluate
    return super().evaluate(eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
  File "/home/di/anaconda3/lib/python3.7/site-packages/transformers/trainer.py", line 1513, in evaluate
    metric_key_prefix=metric_key_prefix,
  File "/home/di/anaconda3/lib/python3.7/site-packages/transformers/trainer.py", line 1629, in prediction_loop
    for step, inputs in enumerate(dataloader):
  File "/home/di/anaconda3/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 363, in __next__
    data = self._next_data()
  File "/home/di/anaconda3/lib/python3.7/site-packages/torch/utils/data/dataloader.py", line 403, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "/home/di/anaconda3/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/home/di/anaconda3/lib/python3.7/site-packages/transformers/data/data_collator.py", line 121, in __call__
    return_tensors="pt",
  File "/home/di/anaconda3/lib/python3.7/site-packages/transformers/tokenization_utils_base.py", line 2641, in pad
    "You should supply an encoding or a list of encodings to this method"
ValueError: You should supply an encoding or a list of encodings to this methodthat includes input_ids, but you provided []
'''
print('enter training')

trainer.train()


print(trainer.evaluate())
print(trainer.evaluate(num_beams=2))

#C:\Users\El\.cache\huggingface\datasets\squad\plain_text\1.0.0\4