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
from torch.autograd import Variable

from .operator_basic import *

_bias = False
_inplace = True

### Conv_Norm
class Conv_Norm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 groups=1, drop_rate=0.0,
                 norm_name='BatchNorm2d', norm_groups=0, norm_k=0, norm_attention_mode=0,
                 replace_stride_with_avgpool=False):
        super(Conv_Norm, self).__init__()

        layers = []
        if stride > 1 and replace_stride_with_avgpool:
            layers.append(nn.AvgPool2d(kernel_size=(stride, stride), stride=stride))
            stride = 1
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding,
                                groups=groups, bias=_bias))
        layers.append(FeatureNorm(norm_name, out_channels, norm_groups, norm_k, norm_attention_mode))
        if drop_rate > 0.0:
            layers.append(nn.Dropout2d(p=drop_rate, inplace=_inplace))
        self.conv_norm = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv_norm(x)
        return y

### Conv_Norm_AC
class Conv_Norm_AC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                groups=1, drop_rate=0., ac_mode=0,
                norm_name='BatchNorm2d', norm_groups=0, norm_k=0, norm_attention_mode=0,
                replace_stride_with_avgpool=False):
        super(Conv_Norm_AC, self).__init__()

        self.conv_norm = Conv_Norm(in_channels, out_channels, kernel_size, stride, padding,
                                groups=groups, drop_rate=drop_rate,
                                norm_name=norm_name, norm_groups=norm_groups, norm_k=norm_k, norm_attention_mode=norm_attention_mode,
                                replace_stride_with_avgpool=replace_stride_with_avgpool)
        self.ac = AC(ac_mode)

    def forward(self, x):
        y = self.conv_norm(x)
        y = self.ac(y)
        return y

### NodeOpSingleScale
class NodeOpSingleScale(nn.Module):
    def __init__(self, in_channels, out_channels, stride,
                groups=1, drop_rate=0., ac_mode=0, bn_ratio=0.25,
                norm_name='BatchNorm2d', norm_groups=0, norm_k=0, norm_attention_mode=0,
                replace_stride_with_avgpool=True):
        super(NodeOpSingleScale, self).__init__()
        if "BatchNorm2d" in norm_name:
            norm_name_base = "BatchNorm2d"
        elif "GroupNorm" in norm_name:
            norm_name_base = "GroupNorm"
        else:
            raise ValueError("Unknown norm layer")

        mid_channels = max(4, to_int(out_channels * bn_ratio / groups) * groups)
        self.conv_norm_ac_1 = Conv_Norm_AC(in_channels, mid_channels, 1,  1, 0,
                                         ac_mode=ac_mode,
                                         norm_name=norm_name_base, norm_groups=norm_groups, norm_k=norm_k, norm_attention_mode=norm_attention_mode)
        self.conv_norm_ac_2 = Conv_Norm_AC(mid_channels, mid_channels, 3, stride, 1,
                                         groups=groups, ac_mode=ac_mode,
                                         norm_name=norm_name, norm_groups=norm_groups, norm_k=norm_k, norm_attention_mode=norm_attention_mode,
                                         replace_stride_with_avgpool=False)
        self.conv_norm_3    = Conv_Norm(mid_channels, out_channels, 1, 1, 0,
                                         drop_rate=drop_rate,
                                         norm_name=norm_name_base, norm_groups=norm_groups, norm_k=norm_k, norm_attention_mode=norm_attention_mode)

        self.shortcut = None
        if in_channels != out_channels or stride > 1:
            self.shortcut = Conv_Norm(in_channels, out_channels, 1, stride, 0,
                                    norm_name=norm_name_base, norm_groups=norm_groups, norm_k=norm_k, norm_attention_mode=norm_attention_mode,
                                    replace_stride_with_avgpool=replace_stride_with_avgpool)

        self.ac = AC(ac_mode)

    def forward(self, x, res=None):
        residual = x if res is None else res
        y = self.conv_norm_ac_1(x)
        y = self.conv_norm_ac_2(y)
        y = self.conv_norm_3(y)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        y += residual
        y = self.ac(y)
        return y

### TODO: write a unit test for NodeOpSingleScale in a standalone way


