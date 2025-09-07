from copy import deepcopy
import numpy as np
import torch
import gc
import random
from lib.quantizer.uniform import Quantizer, dequantize, quantize_dequantize, quantize
from lib.utils.modelutils import find_sublayers, get_layers
from lib.utils.packutils import pack_tensor, unpack_tensor
from lib.utils.perplexity import eval_ppl
from lib.utils.bfautils import generate_error_matrix

@torch.no_grad()
def qerr_inject_awq(W, args, ber, seed):
    dev = W.device
    shape_ = W.shape
    if args.groupsize_w > 0:
        W = W.reshape(-1, args.groupsize_w)
    
    grp_shape = W.shape

    quantizer = Quantizer()
    quantizer.configure(args.bits_w, perchannel=args.perchannel, sym=args.sym_w)
    quantizer.find_params(W, weight=True)

    scale = quantizer.scale.to(dev)
    zero = quantizer.zero.to(dev)
    maxq = quantizer.maxq.to(dev)

    num_weights_per_32bit = 32 // args.bits_w
    packed_tensor_row = W.shape[0] // num_weights_per_32bit

    err_shape = (packed_tensor_row, W.shape[1])
    err_mat = generate_error_matrix(
        shape=err_shape, rate=ber, seed=seed
    ).reshape(err_shape)

    qw = quantize(W, scale, zero, maxq)
    dqw_orig = quantize_dequantize(W, scale, zero, maxq)
    dqw_orig = dqw_orig.reshape(shape_)

    packed_qw, _ = pack_tensor(qw.to(torch.int8), args.bits_w)
    qw_unpack = unpack_tensor(
        packed_qw ^ err_mat.reshape(-1), args.bits_w, grp_shape
    )
    return dequantize(qw_unpack, scale, zero).reshape(shape_)
    

@torch.no_grad()
def qerr_injection(model, quantizers, args, ber, seed):
    random.seed(seed)
    layers = get_layers(model)

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_sublayers(layer)
        for name in subset:
            if args.quant_method == 'awq':
                W = subset[name].weight.data.clone().detach().cuda()
                subset[name].weight.data = qerr_inject_awq(W, args, ber, seed).half()

def model_error_injection(qmodel, quantizers, args, dev):
    ber_range = np.concatenate(
        (np.linspace(1, 9, 9) * 1e-4, np.linspace(1, 10, 10) * 1e-3)
    )
    seed_range = [i for i in range(10)]

    print('Bit error injection loop started')
    for ber in ber_range:
        print(f"BER = {ber}")
        for seed in seed_range:
            print(f"Seed = {seed}")
            print("Make dummy model and quantizers")
            cp_model = deepcopy(qmodel)
            cp_quantizers = deepcopy(quantizers)

            print('Start error injection procedure')
            qerr_injection(cp_model, cp_quantizers, args, ber, seed)
            
            print('Finish error injection')
            print(f'ppl after bit error injection: {eval_ppl(cp_model.to(dev), args, nsamples=50)}')

            del cp_model, cp_quantizers
            gc.collect()
            torch.cuda.empty_cache()
    
