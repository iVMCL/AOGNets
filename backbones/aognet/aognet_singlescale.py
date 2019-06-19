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

import scipy.stats as stats

from .config import cfg
from .AOG import *
from .operator_basic import *
from .operator_singlescale import *

### AOG building block
class AOGBlock(nn.Module):
    def __init__(self, stage, block, aog, in_channels, out_channels, drop_rate, stride):
        super(AOGBlock, self).__init__()
        self.stage = stage
        self.block = block
        self.aog = aog
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_rate = drop_rate
        self.stride = stride

        self.dim = aog.param.grid_wd
        self.in_slices = self._calculate_slices(self.dim, in_channels)
        self.out_slices = self._calculate_slices(self.dim, out_channels)

        self.node_set = aog.node_set
        self.primitive_set = aog.primitive_set
        self.BFS = aog.BFS
        self.DFS = aog.DFS

        self.hasLateral = {}
        self.hasDblCnt = {}

        self.primitiveDblCnt = None
        self._set_primitive_dbl_cnt()

        if "BatchNorm2d" in cfg.norm_name:
            self.norm_name_base = "BatchNorm2d"
        elif "GroupNorm" in cfg.norm_name:
            self.norm_name_base = "GroupNorm"
        else:
            raise ValueError("Unknown norm layer")

        self._set_weights_attr()

        self.extra_norm_ac = self._extra_norm_ac()

    def _calculate_slices(self, dim, channels):
        slices = [0] * dim
        for i in range(channels):
            slices[i % dim] += 1
        for d in range(1, dim):
            slices[d] += slices[d - 1]
        slices = [0] + slices
        return slices

    def _set_primitive_dbl_cnt(self):
        self.primitiveDblCnt = [0.0 for i in range(self.dim)]
        for id_ in self.DFS:
            node = self.node_set[id_]
            arr = self.primitive_set[node.rect_idx]
            if node.node_type == NodeType.TerminalNode:
                for i in range(arr.x1, arr.x2+1):
                    self.primitiveDblCnt[i] += node.npaths
        for i in range(self.dim):
            assert self.primitiveDblCnt[i] >= 1.0

    def _create_op(self, node_id, cin, cout, stride, groups=1,
                    keep_norm_base=False, norm_k=0):
        replace_stride = cfg.aognet.replace_stride_with_avgpool
        setattr(self, 'stage_{}_block_{}_node_{}_op'.format(self.stage, self.block, node_id),
                NodeOpSingleScale(cin, cout, stride,
                                    groups=groups, drop_rate=self.drop_rate,
                                    ac_mode=cfg.activation_mode,
                                    bn_ratio=cfg.aognet.bottleneck_ratio,
                                    norm_name=self.norm_name_base if keep_norm_base else cfg.norm_name,
                                    norm_groups=cfg.norm_groups,
                                    norm_k = norm_k,
                                    norm_attention_mode=cfg.norm_attention_mode,
                                    replace_stride_with_avgpool=replace_stride))

    def _set_weights_attr(self):
        for id_ in self.DFS:
            node = self.node_set[id_]
            arr = self.primitive_set[node.rect_idx]
            bn_ratio = cfg.aognet.bottleneck_ratio
            width_per_group = cfg.aognet.width_per_group
            keep_norm_base = arr.Width()<self.dim  #node.node_type == NodeType.TerminalNode #arr.Width()<self.dim #  False
            norm_k = cfg.norm_k[self.stage] # int(cfg.norm_k[self.stage] * arr.Width() / float(self.dim))
            if node.node_type == NodeType.TerminalNode:
                self.hasLateral[node.id] = False
                self.hasDblCnt[node.id] = False
                inplane = self.in_channels if cfg.aognet.terminal_node_no_slice[self.stage] else \
                            self.in_slices[arr.x2 + 1] - self.in_slices[arr.x1]
                outplane = self.out_slices[arr.x2 + 1] - self.out_slices[arr.x1]
                stride = self.stride
                groups = max(1, to_int(outplane * bn_ratio / width_per_group)) \
                     if cfg.aognet.use_group_conv else 1
                self._create_op(node.id, inplane, outplane, stride, groups=groups,
                                keep_norm_base=keep_norm_base, norm_k=norm_k)

            elif node.node_type == NodeType.AndNode:
                plane = self.out_slices[arr.x2 + 1] - self.out_slices[arr.x1]
                stride = 1
                groups = max(1, to_int(plane * bn_ratio / width_per_group)) \
                     if cfg.aognet.use_group_conv else 1
                self.hasLateral[node.id] = False
                self.hasDblCnt[node.id] = False
                for chid in node.child_ids:
                    ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                    if arr.Width() == ch_arr.Width():
                        self.hasLateral[node.id] = True
                        break
                if cfg.aognet.handle_dbl_cnt:
                    for chid in node.child_ids:
                        ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                        if arr.Width() > ch_arr.Width():
                            if node.npaths / self.node_set[chid].npaths != 1.0:
                                self.hasDblCnt[node.id] = True
                                break
                self._create_op(node.id, plane, plane, stride, groups=groups,
                                keep_norm_base=keep_norm_base, norm_k=norm_k)

            elif node.node_type == NodeType.OrNode:
                assert self.node_set[node.child_ids[0]].node_type != NodeType.OrNode
                plane = self.out_slices[arr.x2 + 1] - self.out_slices[arr.x1]
                stride = 1
                groups = max(1, to_int(plane * bn_ratio / width_per_group)) \
                     if cfg.aognet.use_group_conv else 1
                self.hasLateral[node.id] = False
                self.hasDblCnt[node.id] = False
                for chid in node.child_ids:
                    ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                    if self.node_set[chid].node_type == NodeType.OrNode or arr.Width() < ch_arr.Width():
                        self.hasLateral[node.id] = True
                        break
                if cfg.aognet.handle_dbl_cnt:
                    for chid in node.child_ids:
                        ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                        if not (self.node_set[chid].node_type == NodeType.OrNode or arr.Width() < ch_arr.Width()):
                            if node.npaths / self.node_set[chid].npaths != 1.0:
                                self.hasDblCnt[node.id] = True
                                break
                self._create_op(node.id, plane, plane, stride, groups=groups,
                                keep_norm_base=keep_norm_base, norm_k=norm_k)

    def _extra_norm_ac(self):
        return nn.Sequential(FeatureNorm(self.norm_name_base, self.out_channels,
                                cfg.norm_groups, cfg.norm_k[self.stage],
                                cfg.norm_attention_mode),
                            AC(cfg.activation_mode))

    def forward(self, x):
        NodeIdTensorDict = {}

        # handle input x
        tnode_dblcnt = False
        if cfg.aognet.handle_tnode_dbl_cnt and self.in_channels==self.out_channels:
            x_scaled = []
            for i in range(self.dim):
                left, right = self.in_slices[i], self.in_slices[i+1]
                x_scaled.append(x[:, left:right, :, :].div(self.primitiveDblCnt[i]))
            xx = torch.cat(x_scaled, 1)
            tnode_dblcnt = True

        # T-nodes, (hope they will be computed in parallel by pytorch)
        for id_ in self.DFS:
            node = self.node_set[id_]
            op_name = 'stage_{}_block_{}_node_{}_op'.format(self.stage, self.block, node.id)
            if node.node_type == NodeType.TerminalNode:
                arr = self.primitive_set[node.rect_idx]
                right, left = self.in_slices[arr.x2 + 1], self.in_slices[arr.x1]
                tnode_tensor_op = x if cfg.aognet.terminal_node_no_slice[self.stage] else x[:, left:right, :, :] #.contiguous()
                # assert tnode_tensor.requires_grad, 'slice needs to retain grad'
                if tnode_dblcnt:
                    tnode_tensor_res = xx[:, left:right, :, :].mul(node.npaths)
                    tnode_output = getattr(self, op_name)(tnode_tensor_op, tnode_tensor_res)
                else:
                    tnode_output = getattr(self, op_name)(tnode_tensor_op)
                NodeIdTensorDict[node.id] = tnode_output

        # AND- and OR-nodes
        for id_ in self.DFS:
            node = self.node_set[id_]
            arr = self.primitive_set[node.rect_idx]
            op_name = 'stage_{}_block_{}_node_{}_op'.format(self.stage, self.block, node.id)
            if node.node_type == NodeType.AndNode:
                if self.hasDblCnt[node.id]:
                    child_tensor_res = []
                    child_tensor_op = []
                    for chid in node.child_ids:
                        ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                        if arr.Width() > ch_arr.Width():
                            factor = node.npaths / self.node_set[chid].npaths
                            if factor == 1.0:
                                child_tensor_res.append(NodeIdTensorDict[chid])
                            else:
                                child_tensor_res.append(NodeIdTensorDict[chid].mul(factor))
                            child_tensor_op.append(NodeIdTensorDict[chid])

                    anode_tensor_res = torch.cat(child_tensor_res, 1)
                    anode_tensor_op = torch.cat(child_tensor_op, 1)

                    if self.hasLateral[node.id]:
                        ids1 = set(node.parent_ids)
                        num_shared = 0
                        for chid in node.child_ids:
                            ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if arr.Width() == ch_arr.Width():
                                anode_tensor_op  = anode_tensor_op  + NodeIdTensorDict[chid]
                                if len(ids1.intersection(ids2)) == num_shared:
                                    anode_tensor_res = anode_tensor_res + NodeIdTensorDict[chid]

                    anode_output = getattr(self, op_name)(anode_tensor_op, anode_tensor_res)
                else:
                    child_tensor = []
                    for chid in node.child_ids:
                        ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                        if arr.Width() > ch_arr.Width():
                            child_tensor.append(NodeIdTensorDict[chid])

                    anode_tensor_op = torch.cat(child_tensor, 1)

                    if self.hasLateral[node.id]:
                        ids1 = set(node.parent_ids)
                        num_shared =  0
                        for chid in node.child_ids:
                            ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if arr.Width() == ch_arr.Width() and len(ids1.intersection(ids2)) == num_shared:
                                anode_tensor_op = anode_tensor_op + NodeIdTensorDict[chid]

                        anode_tensor_res = anode_tensor_op

                        for chid in node.child_ids:
                            ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if arr.Width() == ch_arr.Width() and len(ids1.intersection(ids2)) > num_shared:
                                anode_tensor_op = anode_tensor_op + NodeIdTensorDict[chid]

                        anode_output = getattr(self, op_name)(anode_tensor_op, anode_tensor_res)
                    else:
                        anode_output = getattr(self, op_name)(anode_tensor_op)

                NodeIdTensorDict[node.id] = anode_output

            elif node.node_type == NodeType.OrNode:
                if self.hasDblCnt[node.id]:
                    factor = node.npaths / self.node_set[node.child_ids[0]].npaths
                    if factor == 1.0:
                        onode_tensor_res = NodeIdTensorDict[node.child_ids[0]]
                    else:
                        onode_tensor_res = NodeIdTensorDict[node.child_ids[0]].mul(factor)
                    onode_tensor_op = NodeIdTensorDict[node.child_ids[0]]
                    for chid in node.child_ids[1:]:
                        if self.node_set[chid].node_type != NodeType.OrNode:
                            ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                            if arr.Width() == ch_arr.Width():
                                factor = node.npaths / self.node_set[chid].npaths
                                if factor == 1.0:
                                    onode_tensor_res = onode_tensor_res + NodeIdTensorDict[chid]
                                else:
                                    onode_tensor_res = onode_tensor_res + NodeIdTensorDict[chid].mul(factor)
                                if cfg.aognet.use_elem_max_for_ORNodes:
                                    onode_tensor_op  = torch.max(onode_tensor_op, NodeIdTensorDict[chid])
                                else:
                                    onode_tensor_op  = onode_tensor_op  + NodeIdTensorDict[chid]

                    if self.hasLateral[node.id]:
                        ids1 = set(node.parent_ids)
                        num_shared = 0
                        for chid in node.child_ids[1:]:
                            ids2 = self.node_set[chid].parent_ids
                            if self.node_set[chid].node_type == NodeType.OrNode and \
                                len(ids1.intersection(ids2)) == num_shared:
                                onode_tensor_res = onode_tensor_res +  NodeIdTensorDict[chid]
                                if cfg.aognet.use_elem_max_for_ORNodes:
                                    onode_tensor_op  = torch.max(onode_tensor_op, NodeIdTensorDict[chid])
                                else:
                                    onode_tensor_op = onode_tensor_op + NodeIdTensorDict[chid]

                        for chid in node.child_ids[1:]:
                            ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if self.node_set[chid].node_type == NodeType.OrNode and \
                                len(ids1.intersection(ids2)) > num_shared:
                                if cfg.aognet.use_elem_max_for_ORNodes:
                                    onode_tensor_op  = torch.max(onode_tensor_op, NodeIdTensorDict[chid])
                                else:
                                    onode_tensor_op = onode_tensor_op + NodeIdTensorDict[chid]
                            elif self.node_set[chid].node_type == NodeType.TerminalNode and \
                                arr.Width() < ch_arr.Width():
                                ch_left = self.out_slices[arr.x1] - self.out_slices[ch_arr.x1]
                                ch_right = self.out_slices[arr.x2 + 1] - self.out_slices[ch_arr.x1]
                                if cfg.aognet.use_elem_max_for_ORNodes:
                                    onode_tensor_op  = torch.max(onode_tensor_op, NodeIdTensorDict[chid][:, ch_left:ch_right, :, :])
                                else:
                                    onode_tensor_op = onode_tensor_op + NodeIdTensorDict[chid][:, ch_left:ch_right, :, :]#.contiguous()

                    onode_output = getattr(self, op_name)(onode_tensor_op, onode_tensor_res)
                else:
                    if cfg.aognet.use_elem_max_for_ORNodes:
                        onode_tensor_op = NodeIdTensorDict[node.child_ids[0]]
                        onode_tensor_res = NodeIdTensorDict[node.child_ids[0]]
                        for chid in node.child_ids[1:]:
                            if self.node_set[chid].node_type != NodeType.OrNode:
                                ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                                if arr.Width() == ch_arr.Width():
                                    onode_tensor_op  = torch.max(onode_tensor_op, NodeIdTensorDict[chid])
                                    onode_tensor_res = onode_tensor_res + NodeIdTensorDict[chid]

                        if self.hasLateral[node.id]:
                            ids1 = set(node.parent_ids)
                            num_shared = 0
                            for chid in node.child_ids[1:]:
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == NodeType.OrNode and \
                                    len(ids1.intersection(ids2)) == num_shared:
                                    onode_tensor_op  = torch.max(onode_tensor_op, NodeIdTensorDict[chid])
                                    onode_tensor_res = onode_tensor_res + NodeIdTensorDict[chid]

                            for chid in node.child_ids[1:]:
                                ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == NodeType.OrNode and \
                                    len(ids1.intersection(ids2)) > num_shared:
                                    onode_tensor_op  = torch.max(onode_tensor_op, NodeIdTensorDict[chid])
                                elif self.node_set[chid].node_type == NodeType.TerminalNode and \
                                    arr.Width() < ch_arr.Width():
                                    ch_left = self.out_slices[arr.x1] - self.out_slices[ch_arr.x1]
                                    ch_right = self.out_slices[arr.x2 + 1] - self.out_slices[ch_arr.x1]
                                    onode_tensor_op  = torch.max(onode_tensor_op, NodeIdTensorDict[chid][:, ch_left:ch_right, :, :])

                            onode_output = getattr(self, op_name)(onode_tensor_op, onode_tensor_res)
                        else:
                            onode_output = getattr(self, op_name)(onode_tensor_op)
                    else:
                        onode_tensor_op = NodeIdTensorDict[node.child_ids[0]]
                        for chid in node.child_ids[1:]:
                            if self.node_set[chid].node_type != NodeType.OrNode:
                                ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                                if arr.Width() == ch_arr.Width():
                                    onode_tensor_op = onode_tensor_op + NodeIdTensorDict[chid]

                        if self.hasLateral[node.id]:
                            ids1 = set(node.parent_ids)
                            num_shared = 0
                            for chid in node.child_ids[1:]:
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == NodeType.OrNode and \
                                    len(ids1.intersection(ids2)) == num_shared:
                                    onode_tensor_op = onode_tensor_op + NodeIdTensorDict[chid]

                            onode_tensor_res = onode_tensor_op

                            for chid in node.child_ids[1:]:
                                ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == NodeType.OrNode and \
                                    len(ids1.intersection(ids2)) > num_shared:
                                    onode_tensor_op = onode_tensor_op + NodeIdTensorDict[chid]
                                elif self.node_set[chid].node_type == NodeType.TerminalNode and \
                                    arr.Width() < ch_arr.Width():
                                    ch_left = self.out_slices[arr.x1] - self.out_slices[ch_arr.x1]
                                    ch_right = self.out_slices[arr.x2 + 1] - self.out_slices[ch_arr.x1]
                                    onode_tensor_op = onode_tensor_op + NodeIdTensorDict[chid][:, ch_left:ch_right, :, :]#.contiguous()

                            onode_output = getattr(self, op_name)(onode_tensor_op, onode_tensor_res)
                        else:
                            onode_output = getattr(self, op_name)(onode_tensor_op)

                NodeIdTensorDict[node.id] = onode_output

        out = NodeIdTensorDict[self.aog.BFS[0]]
        out = self.extra_norm_ac(out) #TODO: Why this? Analyze it in depth
        return out

### AOGNet
class AOGNet(nn.Module):
    def __init__(self, block=AOGBlock):
        super(AOGNet, self).__init__()
        filter_list = cfg.aognet.filter_list
        self.aogs = self._create_aogs()
        self.block = block
        if "BatchNorm2d" in cfg.norm_name:
            self.norm_name_base = "BatchNorm2d"
        elif "GroupNorm" in cfg.norm_name:
            self.norm_name_base = "GroupNorm"
        else:
            raise ValueError("Unknown norm layer")

        if "Mixture" in cfg.norm_name:
            assert len(cfg.norm_k) == len(filter_list)-1 and any(cfg.norm_k), \
                "Wrong mixture component specification (cfg.norm_k)"
        else:
            cfg.norm_k = [0 for i in range(len(filter_list)-1)]

        self.stem = self._stem(filter_list[0])

        self.stage0 = self._make_stage(stage=0, in_channels=filter_list[0], out_channels=filter_list[1])
        self.stage1 = self._make_stage(stage=1, in_channels=filter_list[1], out_channels=filter_list[2])
        self.stage2 = self._make_stage(stage=2, in_channels=filter_list[2], out_channels=filter_list[3])
        self.stage3 = None
        outchannels = filter_list[3]
        if cfg.dataset == 'imagenet':
            self.stage3 = self._make_stage(stage=3, in_channels=filter_list[3], out_channels=filter_list[4])
            outchannels = filter_list[4]

        self.conv_head = None
        if any(cfg.aognet.out_channels):
            assert len(cfg.aognet.out_channels) == 2
            self.conv_head = nn.Sequential(Conv_Norm_AC(outchannels, cfg.aognet.out_channels[0], 1, 1, 0,
                                            ac_mode=cfg.activation_mode,
                                            norm_name=self.norm_name_base,
                                            norm_groups=cfg.norm_groups,
                                            norm_k=cfg.norm_k[-1],
                                            norm_attention_mode=cfg.norm_attention_mode),
                                            nn.AdaptiveAvgPool2d((1, 1)),
                                            Conv_Norm_AC(cfg.aognet.out_channels[0], cfg.aognet.out_channels[1], 1, 1, 0,
                                                ac_mode=cfg.activation_mode,
                                                norm_name=self.norm_name_base,
                                                norm_groups=cfg.norm_groups,
                                                norm_k=cfg.norm_k[-1],
                                                norm_attention_mode=cfg.norm_attention_mode)
                                            )
            outchannels = cfg.aognet.out_channels[1]
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(outchannels, cfg.num_classes)

        ## initialize
        self._init_params()

    def _stem(self, cout):
        layers = []
        if cfg.dataset == 'imagenet':
            if cfg.stem.imagenet_head7x7:
                layers.append( Conv_Norm_AC(3, cout, 7, 2, 3,
                                            ac_mode=cfg.activation_mode,
                                            norm_name=self.norm_name_base,
                                            norm_groups=cfg.norm_groups,
                                            norm_k=cfg.norm_k[0],
                                            norm_attention_mode=cfg.norm_attention_mode) )
            else:
                plane = cout // 2
                layers.append( Conv_Norm_AC(3,     plane, 3, 2, 1,
                                            ac_mode=cfg.activation_mode,
                                            norm_name=self.norm_name_base,
                                            norm_groups=cfg.norm_groups,
                                            norm_k=cfg.norm_k[0],
                                            norm_attention_mode=cfg.norm_attention_mode) )
                layers.append( Conv_Norm_AC(plane, plane, 3, 1, 1,
                                            ac_mode=cfg.activation_mode,
                                            norm_name=self.norm_name_base,
                                            norm_groups=cfg.norm_groups,
                                            norm_k=cfg.norm_k[0],
                                            norm_attention_mode=cfg.norm_attention_mode) )
                layers.append( Conv_Norm_AC(plane, cout,  3, 1, 1,
                                            ac_mode=cfg.activation_mode,
                                            norm_name=self.norm_name_base,
                                            norm_groups=cfg.norm_groups,
                                            norm_k=cfg.norm_k[0],
                                            norm_attention_mode=cfg.norm_attention_mode) )
            if cfg.stem.replace_maxpool_with_res_bottleneck:
                layers.append( NodeOpSingleScale(cout, cout, 2,
                                    ac_mode=cfg.activation_mode,
                                    bn_ratio=cfg.aognet.bottleneck_ratio,
                                    norm_name=self.norm_name_base,
                                    norm_groups=cfg.norm_groups,
                                    norm_k = cfg.norm_k[0],
                                    norm_attention_mode=cfg.norm_attention_mode,
                                    replace_stride_with_avgpool=True) ) # used in OctConv
            else:
                layers.append( nn.MaxPool2d(2, 2) )
        elif cfg.dataset == 'cifar10' or cfg.dataset == 'cifar100':
            layers.append( Conv_Norm_AC(3, cout, 3, 1, 1,
                                        ac_mode=cfg.activation_mode,
                                        norm_name=self.norm_name_base,
                                        norm_groups=cfg.norm_groups,
                                        norm_k=cfg.norm_k[0],
                                        norm_attention_mode=cfg.norm_attention_mode) )
        else:
            raise NotImplementedError

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if cfg.init_mode == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif cfg.init_mode == 'avg':
                    n = m.kernel_size[0] * m.kernel_size[1] * (m.in_channels + m.out_channels) / 2
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                else: # cfg.init_mode == 'kaiming': as default
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (MixtureBatchNorm2d, MixtureGroupNorm)): # before BatchNorm2d
                nn.init.normal_(m.weight_, 1, 0.1)
                nn.init.normal_(m.bias_, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        #  handle dbl cnt in init
        if cfg.aognet.handle_dbl_cnt_in_param_init:
            import re
            for name_, m in self.named_modules():
                if 'node' in name_:
                    idx = re.findall(r'\d+', name_)
                    sid = int(idx[0])
                    nid = int(idx[2])
                    npaths = self.aogs[sid].node_set[nid].npaths
                    if npaths > 1:
                        scale = 1.0 / npaths
                        with torch.no_grad():
                            for ch in m.modules():
                                if isinstance(ch, nn.Conv2d):
                                    ch.weight.mul_(scale)

        # TODO: handle zero-gamma in the last norm layer of bottleneck op

    def _create_aogs(self):
        aogs = []
        num_stages = len(cfg.aognet.filter_list) - 1
        for i in range(num_stages):
            grid_ht = 1
            grid_wd = int(cfg.aognet.dims[i])
            aogs.append(get_aog(grid_ht=grid_ht, grid_wd=grid_wd, max_split=cfg.aognet.max_split[i],
                                use_tnode_topdown_connection=             cfg.aognet.extra_node_hierarchy[i] == 1,
                                use_tnode_bottomup_connection_layerwise=  cfg.aognet.extra_node_hierarchy[i] == 2,
                                use_tnode_bottomup_connection_sequential= cfg.aognet.extra_node_hierarchy[i] == 3,
                                use_node_lateral_connection=              cfg.aognet.extra_node_hierarchy[i] == 4,
                                use_tnode_bottomup_connection=            cfg.aognet.extra_node_hierarchy[i] == 5,
                                use_node_lateral_connection_1=            cfg.aognet.extra_node_hierarchy[i] == 6,
                                remove_symmetric_children_of_or_node=cfg.aognet.remove_symmetric_children_of_or_node[i]
                                ))

        return aogs

    def _make_stage(self, stage, in_channels, out_channels):
        blocks = nn.Sequential()
        dim = cfg.aognet.dims[stage]
        assert in_channels % dim == 0 and out_channels % dim == 0
        step_channels = (out_channels - in_channels) // cfg.aognet.blocks[stage]
        if step_channels % dim != 0:
            low = (step_channels // dim) * dim
            high = (step_channels // dim + 1) * dim
            if (step_channels-low) <= (high-step_channels):
                step_channels = low
            else:
                step_channels = high

        aog = self.aogs[stage]
        for j in range(cfg.aognet.blocks[stage]):
            name_ = 'stage_{}_block_{}'.format(stage, j)
            drop_rate = cfg.aognet.drop_rate[stage]
            stride = cfg.aognet.stride[stage] if j==0 else 1
            outchannels = (in_channels + step_channels) if j < cfg.aognet.blocks[stage]-1 else out_channels
            if stride > 1 and cfg.aognet.when_downsample == 1:
                blocks.add_module(name_ + '_transition',
                                nn.Sequential( Conv_Norm_AC(in_channels, in_channels, 1, 1, 0,
                                            ac_mode=cfg.activation_mode,
                                            norm_name=self.norm_name_base,
                                            norm_groups=cfg.norm_groups,
                                            norm_k=cfg.norm_k[stage],
                                            norm_attention_mode=cfg.norm_attention_mode,
                                            replace_stride_with_avgpool=False),
                                            nn.AvgPool2d(kernel_size=(stride, stride), stride=stride)
                                            )
                            )
                stride = 1
            elif (stride > 1 or in_channels != outchannels) and cfg.aognet.when_downsample == 2:
                trans_op = [Conv_Norm_AC(in_channels, outchannels, 1, 1, 0,
                                            ac_mode=cfg.activation_mode,
                                            norm_name=self.norm_name_base,
                                            norm_groups=cfg.norm_groups,
                                            norm_k=cfg.norm_k[stage],
                                            norm_attention_mode=cfg.norm_attention_mode,
                                            replace_stride_with_avgpool=False)]
                if stride > 1:
                    trans_op.append(nn.AvgPool2d(kernel_size=(stride, stride), stride=stride))
                blocks.add_module(name_ + '_transition', nn.Sequential(*trans_op))
                stride = 1
                in_channels = outchannels

            blocks.add_module(name_, self.block(stage, j, aog, in_channels, outchannels, drop_rate, stride))
            in_channels = outchannels

        return blocks

    def forward(self, x):
        y = self.stem(x)

        y = self.stage0(y)
        y = self.stage1(y)
        y = self.stage2(y)
        if self.stage3 is not None:
            y = self.stage3(y)
        if self.conv_head is not None:
            y = self.conv_head(y)
        else:
            y = self.avgpool(y)
        y = y.view(y.size(0), -1)
        y = self.fc(y)

        return y

def aognet_singlescale(**kwargs):
    '''
    Construct a single scale AOGNet model
    '''
    return AOGNet(**kwargs)
