""" RESEARCH ONLY LICENSE
Copyright (c) 2018-2019 North Carolina State University.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
1. Redistributions and use are permitted for internal research purposes only, and commercial use
is strictly prohibited under this license. Inquiries regarding commercial use should be directed to the
Office of Research Commercialization at North Carolina State University, 919-215-7199,
https://research.ncsu.edu/commercialization/contact/, commercialization@ncsu.edu .
2. Commercial use means the sale, lease, export, transfer, conveyance or other distribution to a
third party for financial gain, income generation or other commercial purposes of any kind, whether
direct or indirect. Commercial use also means providing a service to a third party for financial gain,
income generation or other commercial purposes of any kind, whether direct or indirect.
3. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
4. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.
5. The names “North Carolina State University”, “NCSU” and any trade-name, personal name,
trademark, trade device, service mark, symbol, image, icon, or any abbreviation, contraction or
simulation thereof owned by North Carolina State University must not be used to endorse or promote
products derived from this software without prior written permission. For written permission, please
contact trademarks@ncsu.edu.
Disclaimer: THIS SOFTWARE IS PROVIDED “AS IS” AND ANY EXPRESSED OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NORTH CAROLINA STATE UNIVERSITY BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function  # force to use print as function print(args)
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

_inplace = True
_norm_eps = 1e-5

def to_int(x):
    if x - int(x) < 0.5:
        return int(x)
    else:
        return int(x) + 1

### Activation
class AC(nn.Module):
    def __init__(self, mode):
        super(AC, self).__init__()
        if mode == 1:
            self.ac = nn.LeakyReLU(inplace=_inplace)
        elif mode == 2:
            self.ac = nn.ReLU6(inplace=_inplace)
        else:
            self.ac = nn.ReLU(inplace=_inplace)

    def forward(self, x):
        x = self.ac(x)
        return x

### From mobilenet_v3
class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

### Feature Norm
def FeatureNorm(norm_name, num_channels, num_groups, num_k, attention_mode):
    if norm_name == "BatchNorm2d":
        return nn.BatchNorm2d(num_channels, eps=_norm_eps)
    elif norm_name == "GroupNorm":
        assert num_groups > 1
        if num_channels % num_groups != 0:
            raise ValueError("channels {} not dividable by groups {}".format(num_channels, num_groups))
        return nn.GroupNorm(num_channels, num_groups, eps=_norm_eps)
    elif norm_name == "MixtureBatchNorm2d":
        assert num_k > 1
        return MixtureBatchNorm2d(num_channels, num_k, attention_mode)
    elif norm_name == "MixtureGroupNorm":
        assert num_groups > 1 and num_k > 1
        if num_channels % num_groups != 0:
            raise ValueError("channels {} not dividable by groups {}".format(num_channels, num_groups))
        return MixtureGroupNorm(num_channels, num_groups, num_k, attention_mode)
    else:
        raise NotImplementedError("Unknown feature norm name")

### Attention weights for mixture norm
class AttentionWeights(nn.Module):
    expansion = 2
    def __init__(self, attention_mode, num_channels, k,
                norm_name=None, norm_groups=0):
        super(AttentionWeights, self).__init__()
        self.k = k
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        layers = []
        if attention_mode == 0:
            layers = [ nn.Conv2d(num_channels, k, 1),
                        nn.Sigmoid() ]
        elif attention_mode == 1:
            layers = [ nn.Conv2d(num_channels, k*self.expansion, 1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(k*self.expansion, k, 1),
                        nn.Sigmoid() ]
        elif attention_mode == 2:
            assert norm_name is not None
            layers = [ nn.Conv2d(num_channels, k, 1, bias=False),
                        FeatureNorm(norm_name, k, norm_groups, 0, 0),
                        hsigmoid() ]
        elif attention_mode == 3:
            assert norm_name is not None
            layers = [ nn.Conv2d(num_channels, k*self.expansion, 1, bias=False),
                        FeatureNorm(norm_name, k*self.expansion, norm_groups, 0, 0),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(k*self.expansion, k, 1, bias=False),
                        FeatureNorm(norm_name, k, norm_groups, 0, 0),
                        hsigmoid() ]
        else:
            raise NotImplementedError("Unknow attention weight type")
        self.attention = nn.Sequential(*layers)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x)#.view(b, c)
        return self.attention(y).view(b, self.k)


### Mixture Norm
# TODO: keep it to use FP32 always, need to figure out how to set it using apex ?
class MixtureBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_channels, k, attention_mode, eps=_norm_eps, momentum=0.1,
                 track_running_stats=True):
        super(MixtureBatchNorm2d, self).__init__(num_channels, eps=eps,
            momentum=momentum, affine=False, track_running_stats=track_running_stats)
        self.k = k
        self.weight_ = nn.Parameter(torch.Tensor(k, num_channels))
        self.bias_ = nn.Parameter(torch.Tensor(k, num_channels))

        self.attention_weights = AttentionWeights(attention_mode, num_channels, k,
                                    norm_name='BatchNorm2d')

        self._init_params()

    def _init_params(self):
        nn.init.normal_(self.weight_, 1, 0.1)
        nn.init.normal_(self.bias_, 0, 0.1)

    def forward(self, x):
        output = super(MixtureBatchNorm2d, self).forward(x)
        size = output.size()
        y = self.attention_weights(x) # bxk # or use output as attention input

        weight = y @ self.weight_ # bxc
        bias = y @ self.bias_ # bxc
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias


# Modified on top of nn.GroupNorm
# TODO: keep it to use FP32 always, need to figure out how to set it using apex ?
class MixtureGroupNorm(nn.Module):
    __constants__ = ['num_groups', 'num_channels', 'k', 'eps', 'weight',
                     'bias']

    def __init__(self, num_channels, num_groups, k, attention_mode, eps=_norm_eps):
        super(MixtureGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.k = k
        self.eps = eps
        self.affine = True
        self.weight_ = nn.Parameter(torch.Tensor(k, num_channels))
        self.bias_ = nn.Parameter(torch.Tensor(k, num_channels))
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)

        self.attention_weights = AttentionWeights(attention_mode, num_channels, k,
                                    norm_name='GroupNorm', norm_groups=1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_, 1, 0.1)
        nn.init.normal_(self.bias_, 0, 0.1)

    def forward(self, x):
        output = F.group_norm(
            x, self.num_groups, self.weight, self.bias, self.eps)
        size = output.size()

        y = self.attention_weights(x) # TODO: use output as attention input

        weight = y @ self.weight_
        bias = y @ self.bias_

        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)




