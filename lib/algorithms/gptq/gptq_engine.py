import math
import time

import torch
import torch.nn as nn
import transformers

from lib.quantizer.uniform import quantize_dequantize, quantize, dequantize

DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, thermometer=False, pairing=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        groups = []
        # if static_groups:
        #     import copy
        #     groups = []
        #     for i in range(0, self.columns, groupsize):
        #         quantizer = copy.deepcopy(self.quantizer)
        #         quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
        #         groups.append(quantizer)

        if actorder:
            if pairing:
                desc = torch.argsort(torch.diag(H), descending=True)
                n = desc.size(0)
                perm = torch.empty(n, dtype=torch.long)

                for i in range(n // 2):
                    perm[2 * i] = desc[i]
                    perm[2 * i + 1] = desc[n - 1 - i]
                
                if n % 2 == 1:
                    assert NotImplementedError
                
                W = W[:, perm]
                H = H[perm][:, perm]

                invperm = torch.argsort(perm)

            else:
                perm = torch.argsort(torch.diag(H), descending=True)
                W = W[:, perm]
                H = H[perm][:, perm]
                invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if not thermometer:
                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        #if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            from copy import deepcopy
                            quantizer = deepcopy(self.quantizer)
                            quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                            groups.append(quantizer)
                            self.quantizer = quantizer
                    else:
                        groups.append(self.quantizer)
                        # else:
                        #     idx = i1 + i
                        #     if actorder:
                        #         idx = perm[idx]
                        #     self.quantizer = groups[idx // groupsize]

                    q = quantize_dequantize(
                        w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1
            else:
                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]
                    
                    if groupsize != -1:
                        #if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            from copy import deepcopy
                            quantizer = deepcopy(self.quantizer)
                            quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                            groups.append(quantizer)
                            self.quantizer = quantizer
                    else:
                        groups.append(self.quantizer)
                        # else:
                        #     idx = i1 + i
                        #     if actorder:
                        #         idx = perm[idx]
                        #     self.quantizer = groups[idx // groupsize]

                    if i % 2 == 0:
                        q = quantize_dequantize(
                            w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                        ).flatten()
                    else:
                        # 벡터화된 thermometer branch 처리
                        q_int = quantize(
                            w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                        ).flatten()
                        prev_int = quantize(
                            W1[:, i-1].unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                        ).flatten()

                        diff = q_int - prev_int
                        mask = (diff.abs() < 2)

                        if mask.any():
                            # diff가 0인 경우
                            mask0 = mask & (diff == 0)
                            if mask0.any():
                                q_int[mask0] = torch.where(
                                    q_int[mask0] + 2 > self.quantizer.maxq,
                                    q_int[mask0] - 2,
                                    q_int[mask0] + 2,
                                )
                            # diff가 1인 경우
                            mask1 = mask & (diff == 1)
                            if mask1.any():
                                q_int[mask1] = torch.where(
                                    q_int[mask1] + 1 > self.quantizer.maxq,
                                    q_int[mask1] - 3,
                                    q_int[mask1] + 1,
                                )
                            # diff가 -1인 경우
                            mask_neg1 = mask & (diff == -1)
                            if mask_neg1.any():
                                q_int[mask_neg1] = torch.where(
                                    q_int[mask_neg1] - 1 < 0,
                                    q_int[mask_neg1] + 3,
                                    q_int[mask_neg1] - 1,
                                )

                        q_int = q_int.unsqueeze(1)
                        q = dequantize(q_int, self.quantizer.scale, self.quantizer.zero).flatten()
        
                    
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d ** 2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print(f'time: {(time.time() - tick):.2f}')
        print(f'error: {torch.sum(Losses).item()}')

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
        self.groups = groups

        if actorder:
            return perm, invperm

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
