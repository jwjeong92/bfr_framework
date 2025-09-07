import os
from lib.utils.argutils import parsing_arguments
from lib.utils.perplexity import eval_ppl

args = parsing_arguments()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.num_gpu)

import torch
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from lib.apply_quant import quantize_model
from lib.utils.import_model import model_from_hf_path
from lib.utils.ldstutils import get_save_path, load_quant_model, quant_and_save
from lib.inject_bit_error import model_error_injection


def main(args):
    dev = "cuda"
    save_path = get_save_path(args)

    if os.path.exists(save_path) and not args.force_requant:
        print("Load exist model from disk ...")
        qmodel, tokenizer, quantizers = load_quant_model(save_path)
    else:
        print("Quantization required ...")
        qmodel, tokenizer, quantizers = quant_and_save(save_path, args, dev)

    print("Complete model loading")

    # Check accuracy
    if qmodel.device == 'cpu':
        qmodel = qmodel.to(dev)

    ppls = eval_ppl(qmodel, args, nsamples=50)
    print(f'Baseline Accuracy Evaluation: {ppls}')

    print("Migrate model to CPU")
    qmodel.cpu()

    if args.enable_bf:
        model_error_injection(qmodel, quantizers, args, dev)

if __name__ == "__main__":
    print(args)
    main(args)
