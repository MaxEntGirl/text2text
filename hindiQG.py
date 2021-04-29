# %% [markdown]
# This notebook is an example on how to fine tune mT5 model with Higgingface Transformers to solve multilingual task in 101 lanaguges. This notebook especially takes the problem of question generation in hindi lanagues

# %% [code]
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %% [code]
train = pd.read_csv("../input/xsquad/train.csv")

# %% [code]
train.shape

# %% [code] {"_kg_hide-input":true,"_kg_hide-output":true}
# !pip install transformers
!pip
install
pytorch_lightning == 0.8
.1

# %% [code]
import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# %% [code]
train.head()

# %% [code]
train.shape

# %% [code]
pl.__version__

# %% [code]
pip
install
transformers == 4.0
.0
rc1

# %% [code]
from transformers import (
    AdamW,
    MT5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)


# %% [code]
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

# %% [code]
import pytorch_lightning as pl


# %% [markdown]
# We'll be pytorch-lightning library for training. Most of the below code is adapted from here https://github.com/huggingface/transformers/blob/master/examples/lightning_base.py
#
# The trainer is generic and can be used for any text-2-text task. You'll just need to change the dataset. Rest of the code will stay unchanged for all the tasks.
#
# This is the most intresting and powrfull thing about the text-2-text format. You can fine-tune the model on variety of NLP tasks by just formulating the problem in text-2-text setting. No need to change hyperparameters, learning rate, optimizer or loss function. Just plug in your dataset and you are ready to go!

# %% [code]
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams

        self.model = MT5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return True

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

    def _step(self, batch):
        labels = batch["target_ids"]
        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="valid", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


# %% [code]
logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


# %% [markdown]
# Let's define the hyperparameters and other arguments. You can overide this dict for specific task as needed. While in most of cases you'll only need to change the data_dirand output_dir.
#
# Here the batch size is 8 and gradient_accumulation_steps are 8 so the effective batch size is 64

# %% [code]
args_dict = dict(
    data_dir="",  # path for data files
    output_dir="",  # path to save the checkpoints
    model_name_or_path='google/mt5-base',
    tokenizer_name_or_path='google/mt5-base',
    max_seq_length=512,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=8,
    eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=8,
    n_gpu=1,
    early_stop_callback=False,
    fp_16=False,  # if you want to enable 16-bit training then install apex and set this to true
    opt_level='O1',
    # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    seed=42,
)

# %% [code]
!ls

# %% [code]
!ls.. / input /

# %% [code] {"_kg_hide-output":true}
train_path = "../input/xsquad/train.csv"
val_path = "../input/xsquad/valid.csv"

train = pd.read_csv(train_path)
print(train.head())

# tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')

# %% [code]
df = pd.read_csv(train_path)
df.columns

# %% [code]
df


# %% [code]
class QuestionDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=30):
        self.path = os.path.join(data_dir, type_path + '.csv')

        self.english = 'context'
        self.hindi = 'question'
        self.data = pd.read_csv(self.path)

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}

    def _build(self):
        for idx in range(len(self.data)):
            input_text, output_text = self.data.loc[idx, self.english], self.data.loc[idx, self.hindi]

            input_ = "Hindi Context: %s" % (input_text)
            target = "%s " % (output_text)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_], max_length=200, pad_to_max_length=True, return_tensors="pt"
            )
            # tokenize targets
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target], max_length=20, pad_to_max_length=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


# %% [code]
tokenizer = AutoTokenizer.from_pretrained('google/mt5-large')

# %% [code]
dataset = QuestionDataset(tokenizer, '../input/xsquad/', 'valid', 30)
print("Val dataset: ", len(dataset))

# %% [code]
data = dataset[20]
print(tokenizer.decode(data['source_ids']))
print(tokenizer.decode(data['target_ids']))

# %% [code]
!mkdir
result
!ls
!pwd

# %% [code]
args_dict.update({'data_dir': '../input/xsquad', 'output_dir': '/kaggle/working/result', 'num_train_epochs': 10,
                  'max_seq_length': 220})
args = argparse.Namespace(**args_dict)
print(args_dict)

# %% [code]
checkpoint_callback = pl.callbacks.ModelCheckpoint(

    period=1, filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
)

train_params = dict(
    accumulate_grad_batches=args.gradient_accumulation_steps,
    gpus=args.n_gpu,
    max_epochs=args.num_train_epochs,
    early_stop_callback=False,
    precision=16 if args.fp_16 else 32,
    amp_level=args.opt_level,
    gradient_clip_val=args.max_grad_norm,
    checkpoint_callback=checkpoint_callback,
    callbacks=[LoggingCallback()],
)


# %% [code]
def get_dataset(tokenizer, type_path, args):
    return QuestionDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path,
                           max_len=args.max_seq_length)


# %% [code]
print("Initialize model")
model = T5FineTuner(args)

trainer = pl.Trainer(**train_params)

# %% [code]
print(" Training model")
trainer.fit(model)

print("training finished")

print("Saving model")
model.model.save_pretrained("/kaggle/working/result")

print("Saved model")

# %% [code]
!ls

# %% [code]
1 + 1

# %% [code]
!ls
result

# %% [code]
!pwd

# %% [code]
!cp - r / kaggle / working / result / pytorch_model.bin / kaggle / working /
!cp - r / kaggle / working / result / config.json / kaggle / working /

# %% [code]
!ls


# %% [code]
!rm - rf
result

# %% [code]
!ls

# %% [code]
