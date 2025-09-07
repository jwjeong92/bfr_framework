import torch

import scipy.sparse as sp

from lib.utils.modelutils import find_sublayers, get_layers
from lib.utils.packutils import pack_tensor, unpack_tensor

def generate_error_matrix(shape, rate, seed, dtype=torch.int32):
    if len(shape) == 1:
        N = shape[0]
        M = 1
    else:
        N, M = shape

    if dtype == torch.int16:
        bit_width = 16
    elif dtype == torch.int32:
        bit_width = 32
    else:
        bit_width = 64

    bin_error_np = sp.random(
        N, M*bit_width,
        density=rate,
        dtype=bool,
        random_state=np.random.default_rng(seed)
    ).toarray()

    bin_error = torch.from_numpy(bin_error_np).bool()
    bin_error_3d = bin_error.view(N, M, bit_width)
    bits = (2**torch.arange(bit_width, dtype=torch.int32)).view(1, 1, bit_width)
    int_error_matrix = (bin_error_3d.float()*bits).sum(dim=2).to(dtype)

    return int_error_matrix

@torch.no_grad()
def bf_linears(model, quantizers, bf_mode, ber, seed):

    layers = get_layers(model)
    for i in range(len(layers)):
        layer = layers[i]
        sublayers = find_sublayers(layer)
        for name, sublayer in sublayers.items():
            quantizer = quantizers[name]
            bits = quantizer.bits
            scale = quantizer.scale
            zero = quantizer.zero
            maxq = quantizer.maxq

            weight = sublayer.weight.detach().clone()
            # random, burst 
            # 1) pack, 2) add noise, 3) unpack
            # pack
            packed_weight, w_shape = pack_tensor(weight, bits)
            # add noise

            # unpack
            noisy_weight = unpack_tensor(packed_weight, bits, w_shape)
            sublayer.weight.data = noisy_weight


            

            # bfa
            # 1) find most sensitive weights in current state, 
            # 2) select bit index which flipped
    return
