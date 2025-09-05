import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

def quantize_dequantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q

def dequantize(x, scale, zero):
    return scale * (x - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        # spqr configs
        round_zero: bool = False,
        qq_scale_bits=None,
        qq_zero_bits=None,
        qq_groupsize=16,
        qq_zero_sym=False,
        reserved_bins: int = 0,
        qqq_params=None,
    ):
        self.bits = bits
        self.maxq = torch.tensor(2 ** bits - 1 - reserved_bins)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        self.round_zero = round_zero
        
        self.qq_scale_bits = qq_scale_bits
        self.qq_zero_bits = qq_zero_bits
        self.qq_zero_sym = qq_zero_sym
        self.qq_groupsize = qq_groupsize
        self.qqq_params = qqq_params or {}

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)
        maybe_round_zero = torch.round if self.round_zero else lambda x: x

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if self.qq_scale_bits is not None:
            scale_groups = self.scale.reshape(-1, self.qq_groupsize)
            self.qq_scale = Quantizer(shape=scale_groups.shape)
            self.qq_scale.configure(self.qq_scale_bits, perchannel=True, sym=False, round_zero=False, **self.qqq_params)
            self.qq_scale.find_params(scale_groups, weight=True)
            assert self.qq_scale.scale.shape == (scale_groups.shape[0], 1), self.qq_scale.scale.shape
            self.quant_scale = self.qq_scale.quantize(scale_groups)
            self.scale = self.qq_scale.dequantize(self.quant_scale).reshape_as(self.scale)

        if self.qq_zero_bits is not None and ((not self.round_zero) or self.qq_zero_bits < self.bits):
            zero_groups = self.zero.reshape(-1, self.qq_groupsize)
            self.qq_zero = Quantizer(shape=zero_groups.shape)
            self.qq_zero.configure(
                self.qq_zero_bits, perchannel=True, sym=self.qq_zero_sym, round_zero=False, **self.qqq_params
            )
            self.qq_zero.find_params(zero_groups, weight=True)
            assert self.qq_zero.scale.shape == (zero_groups.shape[0], 1), self.qq_zero.scale.shape
            self.quant_zero = self.qq_zero.quantize(zero_groups)
            self.zero = self.qq_zero.dequantize(self.quant_zero).reshape_as(self.zero)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize_dequantize(self, x):
        if self.ready():
            return quantize_dequantize(x, self.scale, self.zero, self.maxq)
        return x
    
    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def dequantize(self, x):
        if self.ready():
            return dequantize(x, self.scale, self.zero)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


try:
    import quant_cuda
except:
    print('CUDA extension not installed.')

# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class Quant3Linear(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) 

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype
            if self.faster:
                x = x.half()
                quant_cuda.vecquant3matmul_faster(x, self.qweight, y, self.scales, self.zeros)
            else:
                x = x.float()
                quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError('Only supports a single token currently.')

def make_quant3(module, names, name='', faster=False):
    if isinstance(module, Quant3Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant3Linear(tmp.in_features, tmp.out_features, faster=faster)
            )
    for name1, child in module.named_children():
        make_quant3(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1
):
    org_w_shape = w.shape
    org_w_dtype = w.dtype
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    
    quantizer = Quantizer()
    quantizer.configure(n_bit, perchannel=True, sym=False)
    quantizer.find_params(w, weight=True)
    scale = quantizer.scale
    zero = quantizer.zero
    maxq = quantizer.maxq
    w = quantize_dequantize(w, scale, zero, maxq)
    w = w.reshape(org_w_shape).to(org_w_dtype)
    
    return w


# def pseudo_quantize_tensor(
#     w, n_bit=8, zero_point=True,q_group_size=-1, inplace=False, get_scale_zp=False
# ):
#     org_w_shape = w.shape
#     if q_group_size > 0:
#         assert org_w_shape[-1] % q_group_size == 0
#         w = w.reshape(-1, q_group_size)
#     assert w.dim() == 2
#     if zero_point:
#         max_val = w.amax(dim=1, keepdim=True)
#         min_val = w.amin(dim=1, keepdim=True)
#         max_int = 2**n_bit - 1
#         min_int = 0
#         scales = (max_val - min_val).clamp(min=1e-5) / max_int
#         zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
#     else:  # we actually never used this
#         assert min_val is None
#         max_val = w.abs().amax(dim=1, keepdim=True)
#         max_val = max_val.clamp(min=1e-5)
#         max_int = 2 ** (n_bit - 1) - 1
#         min_int = -(2 ** (n_bit - 1))
#         scales = max_val / max_int
#         zeros = 0

#     assert torch.isnan(scales).sum() == 0
#     assert torch.isnan(w).sum() == 0

#     if inplace:
#         (
#             (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
#         ).mul_(scales)
#     else:
#         w = (
#             torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
#         ) * scales
#     assert torch.isnan(w).sum() == 0

#     w = w.reshape(org_w_shape)

#     if get_scale_zp:
#         return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
#     else:
#         return w

def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from lib.utils.modelutils import get_layers, get_named_linears
    
    layers = get_layers(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quatnization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config
            )
            m.cpu()