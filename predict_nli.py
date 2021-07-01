import os
import json
import numpy as np
import torch
import argparse
import transformers
from tqdm import tqdm

from torch.utils.data import (DataLoader, TensorDataset, SequentialSampler)

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed)

from datasets import load_dataset

    

def load_data(args, data_path, tokenizer):
    
    def preprocess(examples):
        task_key = ('premise', 'hypothesis')
        inputs = ((examples[task_key[0]], examples[task_key[1]]))
        result = tokenizer(*inputs, padding=args.padding, max_length=args.max_seq, truncation=True)
        return result
    
    datasets = load_dataset("json", data_files={"test":data_path}, field='data', cache_dir=args.cache_dir) 
    datasets = datasets.map(preprocess, batched=True)['test']

    sampler = SequentialSampler(datasets)
    dataloader = DataLoader(datasets, sampler=sampler, batch_size=args.batch_size)

    return dataloader
   
def predict(args, model, dataloader):
    
    predictions = []
    for step, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            inputs = {'input_ids': torch.transpose(torch.stack(batch['input_ids']), 0, 1),
                       'attention_mask': torch.transpose(torch.stack(batch['attention_mask']), 0, 1)}
            
            if args.cuda:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}

            outputs = model(**inputs)
            preds = torch.argmax(outputs['logits'], dim=1).cpu().detach().tolist()
            predictions += preds

           
    return predictions


def write_result(input_path, output_path, preds):
    
    original = json.load(open(input_path, 'r'))
    labeled = []
    for i, l in enumerate(original['data']):
        l['label'] = preds[i]
        labeled.append(l)
    with open(output_path, 'w') as f:
        json.dump({'data': labeled}, f)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='dataset/sampled/')
    parser.add_argument('--pred-dir', type=str, default='dataset/sampled_preds/')
    parser.add_argument('--cache-dir', type=str, default='/mnt/.cache/huggingface/')
    parser.add_argument('--model_name_or_path', type=str, default='roberta-large-mnli')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--cuda', type=bool, default=True)

    args = parser.parse_args()

    topics = sorted(os.listdir(args.data_dir))
# topics = sorted(os.listdir(args.data_dir))[:1000]
# topics = sorted(os.listdir(args.data_dir))[1000:2000]
    ids = [1, 26, 59, 67, 79, 83, 115, 121, 139, 147, 160, 164, 503, 695, 861, 1621, 1634, 2295, 2461, 2476]
    topics = ['topic_' + str(i) + '.json' for i in ids]
    
    # load_model
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    args.max_seq = tokenizer.model_max_length
    args.padding = 'max_length'


    if args.cuda:
        model = model.to('cuda')
    

    for t in topics:
        print("Start processing ", t)
        t_path = args.data_dir + t
        pred_path = args.pred_dir + t
        try:
            dataloader = load_data(args, t_path, tokenizer)
        except:
            continue
        preds = predict(args, model, dataloader) 
        write_result(t_path, pred_path, preds)
