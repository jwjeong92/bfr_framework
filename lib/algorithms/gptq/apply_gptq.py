import time
import torch

from lib.utils.datautils import get_loaders
from lib.utils.modelutils import (
    get_layers, find_sublayers, get_sequential_groups
)
from lib.calibration.cache_inps import get_inps
from lib.algorithms.gptq.gptq_engine import GPTQ
from lib.quantizer.uniform import Quantizer

@torch.no_grad()
def gptq_sequential(model, dataloader, args, dev):
    print("\nStarting GPTQ quantization...")

    # get_inps in GPTQ
    print("catching inputs from data")

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, args.seqlen, model.config.hidden_size), 
        dtype=dtype, device=dev
    )
    inps, forward_args = get_inps(
        model, dataloader, args,
        dev="cpu" if args.offload_activations else dev
    )

    outs = torch.zeros_like(inps)
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    save = args.save_path

    quantizers = {}

    layers = get_layers(model)

    for i in range(len(layers)):
        normal_outlier_count, w_count = 0, 0
        stats_payload = ()
        start_time = time.time()

        layer_dev_original = next(layers[i].parameters()).device
        if layer_dev_original.type != "cuda":
            layer = layers[i].to(dev)
        else:
            layer = layers[i]
        layer_dev = next(layers[i].parameters()).device
        all_sublayers = find_sublayers(layer)

        for k, v in forward_args.items():
            forward_args[k] = v.to(layer_dev) if isinstance(v, torch.Tensor) else v
            
        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(all_sublayers.keys())]

        for names in sequential:
            subset = {n: all_sublayers[n] for n in names}

            handlers = {}
            for sublayer_name in subset:
                handlers[sublayer_name] = GPTQ(subset[sublayer_name])
                handlers[sublayer_name].quantizer = Quantizer()
                handlers[sublayer_name].quantizer.configure(
                    args.bits_w, perchannel=True, sym=args.sym_w, mse=False
                )
            
            def add_batch(name):
                def tmp(_, inp, out):
                    handlers[name].add_batch(inp[0].data, out.data)
                
                return tmp
            
            handles = []
            for sublayer_name in subset:
                handles.append(subset[sublayer_name].register_forward_hook(add_batch(sublayer_name)))
            for j in range(args.nsamples):
                if 'llama' in args.model_path:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=forward_args['attention_mask'], position_ids=forward_args['position_ids'])[0]
                elif 'opt' in args.model_path:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=forward_args['attention_mask'])[0]
            
                if args.offload_activations:
                    outs[j] = outs[j].cpu()
            for h in handles:
                h.remove()
            
            torch.cuda.empty_cache()
            
            for sublayer_name in subset:
                print(f'Quantizing module {sublayer_name} of layer {i}')
                if args.act_order:
                    perm, invperm = handlers[sublayer_name].fasterquant(
                        percdamp=args.percdamp,
                        groupsize=args.groupsize_w,
                        actorder=args.act_order,
                        # thermometer=args.thermometer if sublayer_name != 'mlp.down_proj' else False,
                        thermometer=args.thermometer,
                        pairing=args.pairing,
                    )
                    quantizers['model.decoder.layers.%d.%s' % (i, sublayer_name)] = (handlers[sublayer_name].groups, perm, invperm)
                else:
                    handlers[sublayer_name].fasterquant(
                        percdamp=args.percdamp,
                        groupsize=args.groupsize_w,
                        actorder=args.act_order,
                        # thermometer=args.thermometer if sublayer_name != 'mlp.down_proj' else False,
                        thermometer=args.thermometer,
                        pairing=args.pairing,
                    )
                    quantizers['model.decoder.layers.%d.%s' % (i, sublayer_name)] = handlers[sublayer_name].groups
                
                handlers[sublayer_name].free()
                 
        
        out_losses = []
        for j in range(args.nsamples):
            if 'llama' in args.model_path:
                outs_batch = layer(inps[j].unsqueeze(0), attention_mask=forward_args['attention_mask'], position_ids=forward_args['position_ids'])[0]
            elif 'opt' in args.model_path:
                outs_batch = layer(inps[j].unsqueeze(0), attention_mask=forward_args['attention_mask'])[0]
            
            if not args.skip_out_loss:
                outs_batch_loss = (
                    (outs_batch - outs[j].to(layer_dev))
                    .float()
                    .square()
                    .view(outs_batch.shape[0], -1)
                    .mean(dim=1)
                    .sqrt()
                )
                outs_batch_loss /= outs_batch.view(outs_batch.shape[0], -1).float().std(dim=1)
                out_losses.append(outs_batch_loss)
            outs[j] = outs_batch
            if args.offload_activations:
                outs[j] = outs[j].cpu()
        del outs_batch

        del layer
        del handlers
        torch.cuda.empty_cache()
        
        inps, outs = outs, inps

        print(f'Out losses = {torch.mean(torch.Tensor(out_losses)).item()}')

    model.config.use_cache = use_cache
    
    return quantizers

# inplace-weight update method
def quantize_gptq(model, args, dev):
    dataloader = get_loaders(
        args.dataset, nsamples=args.nsamples,
        seed=args.dset_seed, model=args.model_path,
        seqlen=args.seqlen, cache_dir=args.cache_dir,
    )
    quantizers = gptq_sequential(model, dataloader, args, dev)
    return quantizers