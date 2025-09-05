from lib.utils.perplexity import eval_ppl
from lib.utils.ldstutils import load_quant_model, get_save_path
from lib.utils.argutils import parsing_arguments


args = parsing_arguments()
save_path = get_save_path(args)

qmodel, tokenizer, quantizers = load_quant_model(save_path=save_path)

print(f'ppls = {eval_ppl(qmodel, args)}')