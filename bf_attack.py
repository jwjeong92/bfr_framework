from lib.utils.argutils import parsing_arguments
from lib.utils.ldstutils import get_save_path, load_quant_model, load_pretrained

if __name__ == "__main__":
    args = parsing_arguments()
    save_path = get_save_path(args=args)
    qmodel, quantizers, tokenizer = load_quant_model(save_path=save_path)

    from lib.utils.modelutils import get_layers, find_sublayers, get_named_linears
    from lib.utils.bfautils import bf_linears
    from copy import deepcopy

    bf_mode = args.bf_mode
    bers = args.bers
    becs = args.becs
    seed = args.seed
    
    if bf_mode == 'random' or bf_mode == 'burst':
        for ber in bers:
            cp_model = deepcopy(qmodel)
            bf_linears(cp_model, quantizers, bf_mode, ber, seed)
            
    
