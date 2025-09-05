import os
from lib.utils.argutils import parsing_arguments
from lib.utils.ldstutils import get_save_path
args = parsing_arguments()
setattr(args, 'save_path', get_save_path(args))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISUBLE_DEVICES"]=str(args.num_gpu)

from transformers import AutoTokenizer
from lib.utils.ldstutils import (
    get_save_path, 
    load_quant_model, save_quant_model,
    load_pretrained,
)
from lib.apply_quant import quantize_model

def main(args):
    # Prepare the target model
    dev = "cuda"
    save_path = get_save_path(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if os.path.exists(save_path):
        print("Quant model already exists.")
        print("Loading exist model from disk...")
        qmodel, quantizers = load_quant_model(save_path)
    else:
        print("Quantization required ...")
        os.makedirs(save_path, exist_ok=True)
        model = load_pretrained(args, dev)
        qmodel, quantizers = quantize_model(model, args, dev)
        save_quant_model(save_path, qmodel, quantizers, tokenizer)

if __name__ == '__main__':
    print(args)
    main(args)