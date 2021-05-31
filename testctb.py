
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import csv
from transformers import MT5TokenizerFast,MT5ForConditionalGeneration
from trainctb import PostagDataset,T5FineTuner,args


tokenizer = MT5TokenizerFast.from_pretrained('mt5tokenizer')
dataset = PostagDataset(tokenizer, 'ctb', 'test',  max_len=140)
loader = DataLoader(dataset, batch_size=8, num_workers=1)
model = T5FineTuner(args)
model.model=model.model.from_pretrained('ctb/result')
model.model.eval()
outputs = []
targets = []
f1_list=[]


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    device = "cuda:0"
    model = model.to(device)
#    truep, ref, pred = 0, 0, 0
    data=[]
    for batch in tqdm(loader):
        outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                                    attention_mask=batch['source_mask'].cuda(),
                                    max_length=250)

        '''pred = [tokenizer.decode(ids) for ids in outs]
        target = [tokenizer.decode(ids) for ids in batch["target_ids"]]

        outputs.extend(pred)
        targets.extend(target)'''
        predictions = tokenizer.batch_decode(outs,
                                             skip_special_tokens=True)
        references = tokenizer.batch_decode(batch['target_ids'],
                                            skip_special_tokens=True)
        for r,p in zip(references,predictions):
            pr = set(p.split('/'))
            re = set(r.split('/'))
            truep = len(pr.intersection(re))
            ref = len(re)
            pred = len(pr)
            precision = truep / pred
            recall = truep / ref
            if precision == 0 and recall == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)
            data.append([f1,r, p])

        '''for p, r in zip(predictions, references):
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

    print(f1)'''
    print(len(data))
    with open('ctbpred2.csv', "w",encoding='utf-8',newline='') as csvfile:  #
        writer = csv.writer(csvfile)
        writer.writerow(['f1',"ref", "pred"])
        writer.writerows(data)
