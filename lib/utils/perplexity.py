from lib.utils import datautils
import random
import torch
import logging
from tqdm import tqdm

def eval_ppl(model, args,
             nsamples=None, datasets=['wikitext2','c4'],
            ):
    torch.set_grad_enabled(False)
    random.seed(args.dset_seed)
    torch.random.manual_seed(args.dset_seed)

    results = dict()

    for dataset in datasets:
        input_tok = datautils.get_test_tokens(dataset,
                        seed=args.dset_seed,
                        seqlen=args.seqlen,
                        model=args.model_path,
                        cache_dir=args.cache_dir,
                    )
        if nsamples is None:
            nsamples = input_tok.numel() // args.seqlen
        input_tok = input_tok[0, :(args.seqlen * nsamples)].view(
            nsamples, args.seqlen)

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = range(nsamples)
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(input,
                        use_cache=False,
                        output_hidden_states=False,
                        output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            acc_loss += loss.item()
            # progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / nsamples

        ppl = torch.exp(torch.tensor(avg_loss)).item()
        results[dataset] = ppl
        # logging.info(f'{dataset} perplexity: {ppl}')
    return results

def eval_ppl_wo_args(
        model,
        tokenizer,
        nsamples=None, 
        datasets=['wikitext2','c4'],
        seed=0,
        seqlen=2048,
        model_path='/raid/LLM/opt-125m',
        cache_dir='cache'
        ):
    torch.set_grad_enabled(False)
    random.seed(seed)
    torch.random.manual_seed(seed)

    results = dict()

    for dataset in datasets:
        input_tok = datautils.get_test_tokens(dataset,
                        seed=seed,
                        seqlen=seqlen,
                        model=model_path,
                        cache_dir=cache_dir,
                    )
        if nsamples is None:
            nsamples = input_tok.numel() // seqlen
        input_tok = input_tok[0, :(seqlen * nsamples)].view(
            nsamples, seqlen)

        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = range(nsamples)
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(input,
                        use_cache=False,
                        output_hidden_states=False,
                        output_attentions=False)[0]
            shift_logits = output[:, :-1, :].contiguous()
            shift_labels = input[:, 1:]
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            acc_loss += loss.item()
            # progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / nsamples

        ppl = torch.exp(torch.tensor(avg_loss)).item()
        results[dataset] = ppl
        # logging.info(f'{dataset} perplexity: {ppl}')
    return results