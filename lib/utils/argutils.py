import argparse
from lib.utils.common import *

def parsing_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_gpu', type=int, default=0)
    
    parser.add_argument('--model_path', type=str, default='/raid/LLM/opt-125m')
    parser.add_argument('--use_cuda_graph', type=str2bool, default=True)
    parser.add_argument('--quant_method', type=str, default='rtn')
    parser.add_argument('--cache_dir', type=str, default='cache')

    parser.add_argument('--dataset', type=str, default='wikitext2')
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--dset_seed', type=int, default=0)

    parser.add_argument('--offload_activations', type=str2bool, default=False)
    parser.add_argument('--true_sequential', type=str2bool, default=False)

    parser.add_argument('--bits_w', type=int, default=4)
    parser.add_argument('--groupsize_w', type=int, default=-1)
    parser.add_argument('--sym_w', type=str2bool, default=True)
    parser.add_argument('--perchannel_w', type=str2bool, default=False)
    parser.add_argument('--act_order', type=str2bool, default=False)
    parser.add_argument('--percdamp', type=float, default=0.01)

    parser.add_argument('--thermometer', type=str2bool, default=False)
    parser.add_argument('--pairing', type=str2bool, default=False)
    
    parser.add_argument('--skip_out_loss', type=str2bool, default=True)
    args = parser.parse_args()

    return args
