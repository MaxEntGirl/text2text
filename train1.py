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
NUM_EPOCHS = 20
PERCENTILES = (80, 100)

TRAIN_BATCH_SIZE = 1         # maybe more stable update with bigger batch, gradient_accumulation_steps trade off with batch size
EVAL_BATCH_SIZE = 1
WARMUP_STEPS = 200
#WEIGHT_DECAY = 0.01
LOGGING_STEPS = 3000
LEARNING_RATE = 5e-05


# baseline

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

# figure out what errors it makes; constrained beam search; baseline -bert encoder / beginning-inside tag


def reformat_for_postag(example):
    pairs = zip(example['tokens'],example['upos'])
    nl = []
    for p in pairs:
        p = p[0]+'_'+ str(p[1])
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

#model_name = "allenai/unifiedqa-t5-small" # you can specify the model size here
#tokenizer = AutoTokenizer.from_pretrained(model_name)

#看最长的句子是什么鬼
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
    gradient_accumulation_steps=8,
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


print('enter training')

trainer.train()


print(trainer.evaluate())
print(trainer.evaluate(num_beams=2,max_length=500))

#C:\Users\El\.cache\huggingface\datasets\squad\plain_text\1.0.0\4