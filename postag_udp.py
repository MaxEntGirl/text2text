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
sys.path.append('/home/di/Desktop/thesis/')
NUM_EPOCHS = 3 if torch.cuda.is_available() else 1
PERCENTILES = (80, 100)

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
WARMUP_STEPS = 200
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 100
LEARNING_RATE = 5e-05



'''dataset['train'].features:
{'deprel': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
 'deps': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
 'feats': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
 'head': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
 'idx': Value(dtype='string', id=None),
 'lemmas': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
 'misc': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
 'text': Value(dtype='string', id=None),
 'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
 'upos': Sequence(feature=ClassLabel(num_classes=18, names=['NOUN', 'PUNCT', 'ADP', 'NUM', 'SYM', 'SCONJ', 'ADJ', 'PART', 'DET', 'CCONJ', 'PROPN', 'PRON', 'X', '_', 'ADV', 'INTJ', 'VERB', 'AUX'], names_file=None, id=None), length=-1, id=None),
 'xpos': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)}'''


UPOS_NAME = ['NOUN', 'PUNCT', 'ADP', 'NUM', 'SYM', 'SCONJ', 'ADJ', 'PART', 'DET', 'CCONJ', 'PROPN', 'PRON', 'X', '_', 'ADV', 'INTJ', 'VERB', 'AUX']


def reformat_for_seg(example):
    example['tgt_texts'] = '_'.join(example['tokens']) #convert it to a str
    return example


def reformat_for_postag(example):
    pairs = zip(example['tokens'],example['upos'])
    nl = []
    for p in pairs:
        p = p[0]+'_'+UPOS_NAME[p[1]] 
        nl.append(p)
    example['tgt_texts'] = '/'.join(nl)
    return example


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
from IPython import embed; embed()
#model_name = "allenai/unifiedqa-t5-small" # you can specify the model size here
#tokenizer = AutoTokenizer.from_pretrained(model_name)


max_length = get_max_length(tokenizer, dataset['train'], 'text', PERCENTILES[0])
max_target_length = get_max_length(tokenizer, dataset['train'], 'tgt_texts', PERCENTILES[1])


def tokenize(batch): 
    return tokenizer.prepare_seq2seq_batch(src_texts=batch['text'], 
                                         tgt_texts=batch['tgt_texts'], 
                                         max_length=max_length, 
                                         truncation=True,
                                         max_target_length=45,
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
  
    predictions = [{'id': str(i), 'prediction': pred.strip().lower()} \
                 for i, pred in enumerate(predictions)]
    references = [{'id': str(i), 'reference': ref.strip().lower()} \
                for i, ref in enumerate(references)]
    common = Counter(predictions) & Counter(references)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(predictions)
    recall = 1.0 * num_same / len(references)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

    '''metric = load_metric('/mymetric.py')
    metric.add_batch(predictions=predictions, references=references)
    '''



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
    weight_decay=WEIGHT_DECAY,
    logging_dir='./logs/',
    evaluation_strategy="steps",
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    #compute_metrics=ud_metrics
)

#print(trainer.evaluate(num_beams=2))
'''  File "/home/di/Desktop/thesis/postag_udp.py", line 132, in <module>
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