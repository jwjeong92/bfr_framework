import os
import torch
import random
from datasets import load_dataset
from transformers import AutoTokenizer

def get_wikitext2(nsamples, seed, seqlen, model):
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    traindata = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train')
    valdata = load_dataset(
        'allenai/c4',
        data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        split='validation')
    
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model='', cache_dir='cache'):
    os.makedirs(f'{cache_dir}/calibset',exist_ok=True)
    possible_name = f'{model.split("/")[-1]}_{name}_{seed}_{nsamples}_{seqlen}_calib.pt'
    possible_path = f'{cache_dir}/calibset/{possible_name}'
    if os.path.exists(possible_path):
        print(f'Loading dataloader from {possible_path}')
        dataloader = torch.load(possible_path)
    else:
        if 'c4' in name:
            dataloader = get_c4(nsamples, seed, seqlen, model)[0]
        elif 'wikitext2' in name:
            dataloader = get_wikitext2(nsamples, seed, seqlen, model)[0]
        else:
            raise NotImplementedError
        torch.save(dataloader, possible_path)
        print(f'Saving dataloader into {possible_path}')
    return dataloader

def get_test_tokens(name, seed=0, seqlen=2048, model='', cache_dir='cache'):
    os.makedirs(f'{cache_dir}/testset',exist_ok=True)
    possible_name = f'{model.split("/")[-1]}_{name}_{seed}_{seqlen}_test.pt'
    possible_path = f'{cache_dir}/testset/{possible_name}'
    if os.path.exists(possible_path):
        # print(f'Loading testset from {possible_path}')
        testset = torch.load(possible_path)
    else:
        train_samples = 0
        if name == 'wikitext2':
            testset = get_wikitext2(train_samples, seed, seqlen,
                                 model)[1]['input_ids']
        elif name == 'c4':
            testset = get_c4(train_samples, seed, seqlen, model)[1].input_ids
        else:
            raise Exception
        torch.save(testset, possible_path)
        print(f'Saving testset into {possible_path}')
    return testset

def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    elif data == "c4":
        dataset = load_dataset(
        'allenai/c4',
        data_files={'train': 'en/c4-train.00000-of-01024.json.gz'},
        split='train')
    elif data == "wikitext2":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]