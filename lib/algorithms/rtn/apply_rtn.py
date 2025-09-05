from copy import deepcopy
from lib.utils.modelutils import find_sublayers
from lib.quantizer.uniform import Quantizer, quantize_dequantize
# inplace-weight update method
def quantize_nearest(model, args):
    if 'llama' in args.model_path:
        layers = model.model.layers
    elif 'opt' in args.model_path:
        layers = model.model.decoder.layers
    
    quantizers = {}
    for i in range(len(layers)):
        print(f'Quantizing layer {i}')
        #layer = layers[i].to(dev)
        layer = layers[i]
        
        subset = find_sublayers(layer)
        for name in subset:
            quantizer = Quantizer()
            quantizer.configure(
                args.bits_w, perchannel=True, sym=args.sym_w, mse=False
            )
            W = subset[name].weight.data
            shape_ = W.shape
            if args.groupsize_w > 0:
                W = W.reshape(-1, args.groupsize_w)
            quantizer.find_params(W, weight=True)
            qW = quantize_dequantize(
                W, quantizer.scale, quantizer.zero, quantizer.maxq
            ).to(next(iter(layer.parameters())).dtype)
            qW = qW.reshape(shape_)
            subset[name].weight.data = qW
            quantizers[f'model.decoder.layers.{i}.{name}'] = deepcopy(quantizer)
    
    return quantizers