
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn import metrics
import pytorch_lightning as pl
from transformers import MT5TokenizerFast,MT5ForConditionalGeneration
from trainudp import PostagDataset,T5FineTuner,args


tokenizer = MT5TokenizerFast.from_pretrained('mt5tokenizer')
dataset = PostagDataset(tokenizer, 'udp', 'test',  max_len=140)
loader = DataLoader(dataset, batch_size=8, num_workers=1)
model = T5FineTuner(args)
model.model=model.model.from_pretrained('results/checkpoint-9500')
model.model.eval()
outputs = []
targets = []
f1_list=[]

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = "cuda:0"
    model = model.to(device)
    truep, ref, pred = 0, 0, 0
    for batch in tqdm(loader):
        outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                                    attention_mask=batch['source_mask'].cuda(),
                                    max_length=140)

        '''pred = [tokenizer.decode(ids) for ids in outs]
        target = [tokenizer.decode(ids) for ids in batch["target_ids"]]

        outputs.extend(pred)
        targets.extend(target)'''
        predictions = tokenizer.batch_decode(outs,
                                             skip_special_tokens=True)
        references = tokenizer.batch_decode(batch['target_ids'],
                                            skip_special_tokens=True)
          # not batchwise
        for p, r in zip(predictions, references):
            p = set(p.split('/'))
            r = set(r.split('/'))
            tp = len(p.intersection(r))
            truep += tp
            ref += len(r)
            pred += len(p)

    precision = truep / pred
    recall = truep / ref
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    print(f1)

from IPython import embed; embed()
