import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lib.utils.import_model import model_from_hf_path
from lib.apply_quant import quantize_model

def get_save_path(args):
    cache_dir = args.cache_dir
    model_name = args.model_path.split('/')[-1]
    quant_method = args.quant_method
    bits_w = args.bits_w
    groupsize_w = args.groupsize_w
    
    save_path = f'{cache_dir}/{quant_method}/{model_name}-w{bits_w}g{groupsize_w}'
    if args.sym_w:
        save_path = save_path + '-sym'
    else:
        save_path = save_path + '-asym'
    if args.perchannel_w:
        save_path = save_path + '-pch'
    else:
        save_path = save_path + '-nopch'

    return save_path

def load_quant_model(save_path):
    model = AutoModelForCausalLM.from_pretrained(save_path, torch_dtype=torch.float16, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    quantizers = torch.load(save_path + '/quantizers.pt', weights_only=False)
    return model, tokenizer, quantizers

def quant_and_save(save_path, args, dev):
    import os
    os.makedirs(save_path, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.quant_method == 'awq':
        kwargs = {
            "torch_dtype": torch.float16,
            "low_cpu_mem_usage": True,
            "device_map": "auto"
        }
        config = AutoConfig.from_pretrained(args.model_path, 
                                            trust_remote_code=True)
        config.use_cache = False
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            trust_remote_code=True,
            **kwargs,
        )
    else:
        model = model_from_hf_path(
            args.model_path,
            args.use_cuda_graph,
            device_map='auto',
        )

    qmodel, quantizers = quantize_model(model=model, args=args, dev=dev)
    qmodel.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(quantizers, save_path+'/quantizers.pt')
    
    return qmodel, tokenizer, quantizers

def save_quant_model(save_path, qmodel, quantizers, tokenizer):
    qmodel.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    torch.save(quantizers, save_path+'/quantizers.pt')

def load_pretrained(args, dev):
    if args.quant_method=='awq':
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True, "device_map": "auto"}
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config.use_cache = False
        model = AutoModelForCausalLM.from_pretrained(
                    args.model_path, config=config, trust_remote_code=True, **kwargs
                )
    else: 
        model = model_from_hf_path(args.model_path,
                    args.use_cuda_graph,
                    device_map='auto' if dev == None else dev,
                )
    
    return model
