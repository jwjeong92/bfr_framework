# from lib.algorithms.awq.apply_awq import quantize_awq
from lib.algorithms.gptq.apply_gptq import quantize_gptq
from lib.algorithms.rtn.apply_rtn import quantize_nearest
# from lib.algorithms.spqr.apply_spqr import quantize_spqr

from transformers import AutoTokenizer

def quantize_model(model, args, dev):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if args.quant_method == 'gptq':
        quantizers = quantize_gptq(model, args, dev)
    # elif args.quant_method == 'spqr':
    #     quantizers = quantize_spqr(model, args, dev)
    # elif args.quant_method == 'awq':
    #     quantizers = quantize_awq(model, tokenizer, args)
    elif args.quant_method == 'rtn':
        quantizers = quantize_nearest(model, args)
    else:
        assert NotImplementedError
    
    return model, quantizers