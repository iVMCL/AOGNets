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

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function  # force to use print as function print(args)
from __future__ import unicode_literals

from math import ceil, floor
from collections import deque
import numpy as np
import os
import random
import math
import copy


def get_aog(grid_ht, grid_wd, min_size=1, max_split=2,
            not_use_large_TerminalNode=False, turn_off_size_ratio_TerminalNode=1./4.,
            not_use_intermediate_TerminalNode=False,
            use_root_TerminalNode=True, use_tnode_as_alpha_channel=0,
            use_tnode_topdown_connection=False,
            use_tnode_bottomup_connection=False,
            use_tnode_bottomup_connection_layerwise=False,
            use_tnode_bottomup_connection_sequential=False,
            use_node_lateral_connection=False,   # not include T-nodes
            use_node_lateral_connection_1=False, # include T-nodes
            use_super_OrNode=False,
            remove_single_child_or_node=False,
            remove_symmetric_children_of_or_node=0,
            mark_symmetric_syntatic_subgraph = False,
            max_children_kept_for_or=1000):
    aog_param = Param(grid_ht=grid_ht, grid_wd=grid_wd, min_size=min_size, max_split=max_split,
                      not_use_large_TerminalNode=not_use_large_TerminalNode,
                      turn_off_size_ratio_TerminalNode=turn_off_size_ratio_TerminalNode,
                      not_use_intermediate_TerminalNode=not_use_intermediate_TerminalNode,
                      use_root_TerminalNode=use_root_TerminalNode,
                      use_tnode_as_alpha_channel=use_tnode_as_alpha_channel,
                      use_tnode_topdown_connection=use_tnode_topdown_connection,
                      use_tnode_bottomup_connection=use_tnode_bottomup_connection,
                      use_tnode_bottomup_connection_layerwise=use_tnode_bottomup_connection_layerwise,
                      use_tnode_bottomup_connection_sequential=use_tnode_bottomup_connection_sequential,
                      use_node_lateral_connection=use_node_lateral_connection,
                      use_node_lateral_connection_1=use_node_lateral_connection_1,
                      use_super_OrNode=use_super_OrNode,
                      remove_single_child_or_node=remove_single_child_or_node,
                      mark_symmetric_syntatic_subgraph = mark_symmetric_syntatic_subgraph,
                      remove_symmetric_children_of_or_node=remove_symmetric_children_of_or_node,
                      max_children_kept_for_or=max_children_kept_for_or)
    aog = AOGrid(aog_param)
    aog.Create()
    return aog


class NodeType(object):
    OrNode = "OrNode"
    AndNode = "AndNode"
    TerminalNode = "TerminalNode"
    Unknow = "Unknown"


class SplitType(object):
    HorSplit = "Hor"
    VerSplit = "Ver"
    Unknown = "Unknown"


class Param(object):
    """Input parameters for creating an AOG
    """

    def __init__(self, grid_ht=3, grid_wd=3, max_split=2, min_size=1, control_side_length=False,
                 overlap_ratio=0., use_root_TerminalNode=False,
                 not_use_large_TerminalNode=False, turn_off_size_ratio_TerminalNode=0.5,
                 not_use_intermediate_TerminalNode= False,
                 use_tnode_as_alpha_channel=0,
                 use_tnode_topdown_connection=False,
                 use_tnode_bottomup_connection=False,
                 use_tnode_bottomup_connection_layerwise=False,
                 use_tnode_bottomup_connection_sequential=False,
                 use_node_lateral_connection=False,
                 use_node_lateral_connection_1=False,
                 use_super_OrNode=False,
                 remove_single_child_or_node=False,
                 remove_symmetric_children_of_or_node=0,
                 mark_symmetric_syntatic_subgraph=False,
                 max_children_kept_for_or=100):
        self.grid_ht = grid_ht
        self.grid_wd = grid_wd
        self.max_split = max_split  # maximum number of child nodes when splitting an AND-node
        self.min_size = min_size  # minimum side length or minimum area allowed
        self.control_side_length = control_side_length
        self.overlap_ratio = overlap_ratio
        self.use_root_terminal_node = use_root_TerminalNode
        self.not_use_large_terminal_node = not_use_large_TerminalNode
        self.turn_off_size_ratio_terminal_node = turn_off_size_ratio_TerminalNode
        self.not_use_intermediate_TerminalNode = not_use_intermediate_TerminalNode
        self.use_tnode_as_alpha_channel = use_tnode_as_alpha_channel
        self.use_tnode_topdown_connection = use_tnode_topdown_connection
        self.use_tnode_bottomup_connection = use_tnode_bottomup_connection
        self.use_tnode_bottomup_connection_layerwise = use_tnode_bottomup_connection_layerwise
        self.use_node_lateral_connection = use_node_lateral_connection
        self.use_node_lateral_connection_1 = use_node_lateral_connection_1
        self.use_tnode_bottomup_connection_sequential = use_tnode_bottomup_connection_sequential
        assert 1 >= self.use_node_lateral_connection_1 + self.use_node_lateral_connection + \
               self.use_tnode_topdown_connection + self.use_tnode_bottomup_connection + \
               self.use_tnode_bottomup_connection_layerwise + self.use_tnode_bottomup_connection_sequential, \
            'only one type of node hierarchy can be used'
        self.use_super_OrNode = use_super_OrNode
        self.remove_single_child_or_node = remove_single_child_or_node
        self.remove_symmetric_children_of_or_node = remove_symmetric_children_of_or_node #0: not, 1: keep left, 2: keep right
        self.mark_symmetric_syntatic_subgraph = mark_symmetric_syntatic_subgraph # true, only mark the nodes which will be removed based on remove_symmetric_children_of_or_node
        self.max_children_kept_for_or = max_children_kept_for_or  # how many child nodes kept for an OR-node

        self.get_tag()

    def get_tag(self):
        # identifier useful for naming a particular aog
        self.tag = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
            self.grid_wd, self.grid_ht, self.max_split,
            self.min_size, self.control_side_length,
            self.overlap_ratio,
            self.use_root_terminal_node,
            self.not_use_large_terminal_node,
            self.turn_off_size_ratio_terminal_node,
            self.not_use_intermediate_TerminalNode,
            self.use_tnode_as_alpha_channel,
            self.use_tnode_topdown_connection,
            self.use_tnode_bottomup_connection,
            self.use_tnode_bottomup_connection_layerwise,
            self.use_tnode_bottomup_connection_sequential,
            self.use_node_lateral_connection,
            self.use_node_lateral_connection_1,
            self.use_super_OrNode,
            self.remove_single_child_or_node,
            self.remove_symmetric_children_of_or_node,
            self.mark_symmetric_syntatic_subgraph,
            self.max_children_kept_for_or)


class Rect(object):
    """A simple rectangle
    """

    def __init__(self, x1=0, y1=0, x2=0, y2=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def __eq__(self, other):
        """Override the default Equals behavior"""
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))

    def Width(self):
        return self.x2 - self.x1 + 1

    def Height(self):
        return self.y2 - self.y1 + 1

    def Area(self):
        return self.Width() * self.Height()

    def MinLength(self):
        return min(self.Width(), self.Height())

    def IsOverlap(self, other):
        assert isinstance(other, self.__class__)

        x1 = max(self.x1, other.x1)
        x2 = min(self.x2, other.x2)
        if x1 > x2:
            return False

        y1 = max(self.y1, other.y1)
        y2 = min(self.y2, other.y2)
        if y1 > y2:
            return False

        return True

    def IsSame(self, other):
        assert isinstance(other, self.__class__)

        return self.Width() == other.Width() and self.Height() == other.Height()


class Node(object):
    """Types of nodes in an AOG
    AND-node (structural decomposition),
    OR-node (alternative decompositions),
    TERMINAL-node (link to data).
    """

    def __init__(self, node_id=-1, node_type=NodeType.Unknow, rect_idx=-1,
                 child_ids=None, parent_ids=None,
                 split_type=SplitType.Unknown, split_step1=0, split_step2=0, is_symm=False,
                 ancestors_ids=None):
        self.id = node_id
        self.node_type = node_type
        self.rect_idx = rect_idx
        self.child_ids = child_ids if child_ids is not None else []
        self.parent_ids = parent_ids if parent_ids is not None else []
        self.ancestors_ids = ancestors_ids if ancestors_ids is not None else [] # root or-node exlusive
        self.split_type = split_type
        self.split_step1 = split_step1
        self.split_step2 = split_step2

        # some utility variables used in object detection models
        self.on_off = True
        self.out_edge_visited_count = []
        self.which_classes_visited = {}  # key=class_name, val=frequency
        self.npaths = 0.0
        self.is_symmetric = False
        self.has_dbl_counting = False

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            res = ((self.node_type == other.node_type) and (self.rect_idx == other.rect_idx))
            if res:
                if self.node_type != NodeType.AndNode:
                    return True
                else:
                    if self.split_type != SplitType.Unknown:
                        return (self.split_type == other.split_type) and (self.split_step1 == other.split_step1) and \
                               (self.split_step2 == other.split_step2)
                    else:
                        return (set(self.child_ids) == set(other.child_ids))

            return False

        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        """Override the default hash behavior (that returns id or the object)"""
        return hash(tuple(sorted(self.__dict__.items())))


class AOGrid(object):
    """The AOGrid defines a Directed Acyclic And-Or Graph
    which is used to explore/unfold the space of latent structures
    of a grid (e.g., a 7 * 7 grid for a 100 * 200 lattice)
    """

    def __init__(self, param_):
        assert isinstance(param_, Param)
        self.param = param_
        assert self.param.max_split > 1
        self.primitive_set = []
        self.node_set = []
        self.num_TNodes = 0
        self.num_AndNodes = 0
        self.num_OrNodes = 0
        self.DFS = []
        self.BFS = []
        self.node_DFS = {}
        self.node_BFS = {}
        self.OrNodeIdxInBFS = {}
        self.TNodeIdxInBFS = {}

        # for color consistency in viz
        self.TNodeColors = {}

    def _AddPrimitve(self, rect):
        assert isinstance(rect, Rect)

        if rect in self.primitive_set:
            return self.primitive_set.index(rect)

        self.primitive_set.append(rect)

        return len(self.primitive_set) - 1

    def _AddNode(self, node, not_create_if_existed=True):
        assert isinstance(node, Node)

        if node in self.node_set and not_create_if_existed:
            node = self.node_set[self.node_set.index(node)]
            return False, node

        node.id = len(self.node_set)
        if node.node_type == NodeType.AndNode:
            self.num_AndNodes += 1
        elif node.node_type == NodeType.OrNode:
            self.num_OrNodes += 1
        elif node.node_type == NodeType.TerminalNode:
            self.num_TNodes += 1
        else:
            raise NotImplementedError

        self.node_set.append(node)

        return True, node

    def _DoSplit(self, rect):
        assert isinstance(rect, Rect)

        if self.param.control_side_length:
            return rect.Width() >= self.param.min_size and rect.Height() >= self.param.min_size

        return rect.Area() > self.param.min_size

    def _SplitStep(self, sz):
        if self.param.control_side_length:
            return self.param.min_size

        if sz >= self.param.min_size:
            return 1
        else:
            return int(ceil(self.param.min_size / sz))

    def _DFS(self, id, q, visited):
        if visited[id] == 1:
            raise RuntimeError

        visited[id] = 1
        for i in self.node_set[id].child_ids:
            if visited[i] < 2:
                q, visited = self._DFS(i, q, visited)

        if self.node_set[id].on_off:
            q.append(id)

        visited[id] = 2

        return q, visited

    def _BFS(self, id, q, visited):
        q = [id]
        visited[id] = 0 # count indegree
        i = 0
        while i < len(q):
            node = self.node_set[q[i]]
            for j in node.child_ids:
                visited[j] += 1
                if j not in q:
                    q.append(j)

            i += 1

        q = [id]
        i = 0
        while i < len(q):
            node = self.node_set[q[i]]
            for j in node.child_ids:
                visited[j] -= 1
                if visited[j] == 0:
                    q.append(j)
            i += 1

        return q, visited

    def _countPaths(self, s, t, npaths):
        if s.id == t.id:
            return 1.0
        else:
            if not npaths[s.id]:
                rect = self.primitive_set[s.rect_idx]
                #ids1 = set(s.ancestors_ids)
                ids1 = set(s.parent_ids)
                num_shared = 0
                for c in s.child_ids:
                    ch = self.node_set[c]
                    ch_rect = self.primitive_set[ch.rect_idx]
                    #ids2 = ch.ancestors_ids
                    ids2 = ch.parent_ids

                    if s.node_type == NodeType.AndNode and ch.node_type == NodeType.AndNode and \
                       rect.Width() == ch_rect.Width() and rect.Height() == ch_rect.Height():
                        continue
                    if s.node_type == NodeType.OrNode and \
                       ((ch.node_type == NodeType.OrNode) or \
                        (ch.node_type == NodeType.TerminalNode and (rect.Area() < ch_rect.Area()) )):
                        continue

                    npaths[s.id] += self._countPaths(ch, t, npaths)
            return npaths[s.id]

    def _AssignParentIds(self):
        for i in range(len(self.node_set)):
            self.node_set[i].parent_ids = []

        for node in self.node_set:
            for i in node.child_ids:
                self.node_set[i].parent_ids.append(node.id)

        for i in range(len(self.node_set)):
            self.node_set[i].parent_ids = list(set(self.node_set[i].parent_ids))

    def _AssignAncestorsIds(self):
        self._AssignParentIds()

        assert len(self.BFS) == len(self.node_set)
        self.node_set[self.BFS[0]].ancestors_ids = []

        for nid in self.BFS[1:]:
            node = self.node_set[nid]
            rect = self.primitive_set[node.rect_idx]
            ancestors = []
            for pid in node.parent_ids:
                p = self.node_set[pid]
                p_rect = self.primitive_set[p.rect_idx]
                equal_size = rect.Width() == p_rect.Width() and \
                                rect.Height() == p_rect.Height()
                # AND-to-AND lateral path
                if node.node_type == NodeType.AndNode and p.node_type == NodeType.AndNode and \
                    equal_size:
                    continue
                # OR-to-OR/T lateral path
                if node.node_type == NodeType.OrNode and \
                    ((p.node_type == NodeType.OrNode and equal_size) or \
                    (p.node_type == NodeType.TerminalNode and (rect.Area() < p_rect.Area()) )):
                    continue
                for ppid in p.ancestors_ids:
                    if ppid != self.BFS[0] and ppid not in ancestors:
                        ancestors.append(ppid)
                ancestors.append(pid)
            self.node_set[nid].ancestors_ids = list(set(ancestors))

    def _Postprocessing(self, root_id):
        self.DFS = []
        self.BFS = []
        visited = np.zeros(len(self.node_set))
        self.DFS, _ = self._DFS(root_id, self.DFS, visited)
        visited = np.zeros(len(self.node_set))
        self.BFS, _ = self._BFS(root_id, self.BFS, visited)
        self._AssignAncestorsIds()

    def _FindNodeIdWithGivenRect(self, rect, node_type):
        for node in self.node_set:
            if node.node_type != node_type:
                continue
            if rect == self.primitive_set[node.rect_idx]:
                return node.id

        return -1

    def _add_tnode_topdown_connection(self):

        assert self.param.use_root_terminal_node

        prim_type = [self.param.grid_ht, self.param.grid_wd]
        tnode_queue = self.find_node_ids_with_given_prim_type(prim_type)
        assert len(tnode_queue) == 1

        i = 0
        while i < len(tnode_queue):
            id_ = tnode_queue[i]
            node = self.node_set[id_]
            i += 1

            rect = self.primitive_set[node.rect_idx]

            ids = []
            for y in range(0, rect.Height()):
                for x in range(0, rect.Width()):
                    if y == 0 and x == 0:
                        continue
                    prim_type = [rect.Height()-y, rect.Width()-x]
                    ids += self.find_node_ids_with_given_prim_type(prim_type, rect)

            ids = list(set(ids))
            tnode_queue += ids

            for pid in ids:
                if id_ not in self.node_set[pid].child_ids:
                    self.node_set[pid].child_ids.append(id_)

    def _add_onode_topdown_connection(self):
        assert self.param.use_root_terminal_node

        prim_type = [self.param.grid_ht, self.param.grid_wd]
        tnode_queue = self.find_node_ids_with_given_prim_type(prim_type)
        assert len(tnode_queue) == 1

        i = 0
        while i < len(tnode_queue):
            id_ = tnode_queue[i]
            node = self.node_set[id_]
            i += 1

            rect = self.primitive_set[node.rect_idx]

            ids = []
            ids_t = []
            for y in range(0, rect.Height()):
                for x in range(0, rect.Width()):
                    if y == 0 and x == 0:
                        continue
                    prim_type = [rect.Height()-y, rect.Width()-x]
                    ids += self.find_node_ids_with_given_prim_type(prim_type, rect, NodeType.OrNode)
                    ids_t += self.find_node_ids_with_given_prim_type(prim_type, rect)

            ids = list(set(ids))
            ids_t = list(set(ids_t))

            for pid in ids:
                if id_ not in self.node_set[pid].child_ids:
                    self.node_set[pid].child_ids.append(id_)

    def _add_tnode_bottomup_connection(self):
        assert self.param.use_root_terminal_node

        # primitive tnodes
        prim_type = [1, 1]
        t_ids = self.find_node_ids_with_given_prim_type(prim_type)
        assert len(t_ids) == self.param.grid_wd * self.param.grid_ht

        # other tnodes will be converted to and-nodes
        for h in range(1, self.param.grid_ht+1):
            for w in range(1, self.param.grid_wd+1):
                if h == 1 and w == 1:
                    continue
                prim_type = [h, w]
                ids = self.find_node_ids_with_given_prim_type(prim_type)
                for id_ in ids:
                    self.node_set[id_].node_type = NodeType.AndNode
                    node = self.node_set[id_]
                    rect = self.primitive_set[node.rect_idx]
                    prim_type = [1, 1]
                    for y in range(rect.y1, rect.y2+1):
                        for x in range(rect.x1, rect.x2+1):
                            parent_rect = Rect(x, y, x, y)
                            ch_ids = self.find_node_ids_with_given_prim_type(prim_type, parent_rect)
                            assert len(ch_ids) == 1
                            if ch_ids[0] not in self.node_set[id_].child_ids:
                                self.node_set[id_].child_ids.append(ch_ids[0])

    def _add_lateral_connection(self):
        self._add_node_bottomup_connection_layerwise(node_type=NodeType.AndNode, direction=1)
        self._add_node_bottomup_connection_layerwise(node_type=NodeType.OrNode, direction=0)

        if not self.param.use_node_lateral_connection_1:
            return self.BFS[0]

        # or for all or nodes
        for node in self.node_set:
            if node.node_type != NodeType.OrNode:
                continue

            ch_ids = node.child_ids
            numCh = len(ch_ids)

            hasLateral = False
            for id_ in ch_ids:
                if self.node_set[id_].node_type == NodeType.OrNode:
                    hasLateral = True
                    numCh -= 1

            minNumCh = 3 if hasLateral else 2
            if len(ch_ids) < minNumCh:
                continue

            # find t-node child
            ch0 = -1
            for id_ in ch_ids:
                if self.node_set[id_].node_type == NodeType.TerminalNode:
                    ch0 = id_
                    break
            assert ch0 != -1

            added = False
            for id_ in ch_ids:
                if id_ == ch0 or self.node_set[id_].node_type == NodeType.OrNode:
                    continue

                if len(self.node_set[id_].child_ids) == 2 or numCh == 2:
                    assert ch0 not in self.node_set[id_].child_ids
                    self.node_set[id_].child_ids.append(ch0)
                    added = True

            if not added:
                for id_ in ch_ids:
                    if id_ == ch0 or self.node_set[id_].node_type == NodeType.OrNode:
                        continue

                    found = True
                    for id__ in ch_ids:
                        if id_ in self.node_set[id__].child_ids:
                            found = False
                    if found:
                        assert ch0 not in self.node_set[id_].child_ids
                        self.node_set[id_].child_ids.append(ch0)

        return self.BFS[0]

    def _add_node_bottomup_connection_layerwise(self, node_type=NodeType.TerminalNode, direction=0):

        prim_types = []
        for node in self.node_set:
            if node.node_type == node_type:
                rect = self.primitive_set[node.rect_idx]
                p = [rect.Height(), rect.Width()]
                if p not in prim_types:
                    prim_types.append(p)

        change_direction = False

        prim_types.sort()

        for p in prim_types:
            ids = self.find_node_ids_with_given_prim_type(p, node_type=node_type)
            if len(ids) < 2:
                change_direction = True
                continue

            if change_direction:
                 direction = 1 - direction

            yx = np.empty((0, 4 if node_type==NodeType.AndNode else 2), dtype=np.float32)
            for id_ in ids:
                node = self.node_set[id_]
                rect = self.primitive_set[node.rect_idx]

                if node_type == NodeType.AndNode:
                    ch_node = self.node_set[node.child_ids[0]]
                    ch_rect = self.primitive_set[ch_node.rect_idx]
                    if ch_rect.x1 != rect.x1 or ch_rect.y1 != rect.y1:
                        ch_node = self.node_set[node.child_ids[1]]
                        ch_rect = self.primitive_set[ch_node.rect_idx]
                    pos = (rect.y1, rect.x1, ch_rect.y2, ch_rect.x2)
                else:
                    pos = (rect.y1, rect.x1)
                yx = np.vstack((yx, np.array(pos)))

            if node_type == NodeType.AndNode:
                ind = np.lexsort((yx[:, 1], yx[:, 0], yx[:, 3], yx[:, 2]))
            else:
                ind = np.lexsort((yx[:, 1], yx[:, 0]))

            istart = len(ind) - 1 if direction == 0 else 0
            iend = 0 if direction == 0 else len(ind) - 1
            step = -1 if direction == 0 else 1
            for i in range(istart, iend, step):
                id_ = ids[ind[i]]
                chid = ids[ind[i - 1]] if direction==0 else ids[ind[i+1]]
                if chid not in self.node_set[id_].child_ids:
                    self.node_set[id_].child_ids.append(chid)

            if change_direction:
                 direction = 1 - direction
                 change_direction = False

    def _add_tnode_bottomup_connection_sequential(self):

        assert self.param.grid_wd > 1 and self.param.grid_ht == 1

        self._add_node_bottomup_connection_layerwise()

        for node in self.node_set:
            if node.node_type != NodeType.TerminalNode:
                continue
            rect = self.primitive_set[node.rect_idx]
            if rect.Width() == 1:
                continue

            rect1 = copy.deepcopy(rect)
            rect1.x1 += 1
            chid = self._FindNodeIdWithGivenRect(rect1, NodeType.TerminalNode)
            if chid != -1:
                self.node_set[node.id].child_ids.append(chid)

    def _mark_symmetric_subgraph(self):

        for i in self.BFS:
            node = self.node_set[i]

            if node.is_symmetric or node.node_type == NodeType.TerminalNode:
                continue

            if i != self.BFS[0]:
                is_symmetric = True
                for j in node.parent_ids:
                    p = self.node_set[j]
                    if not p.is_symmetric:
                        is_symmetric = False
                        break
                if is_symmetric:
                    self.node_set[i].is_symmetric = True
                    continue

            rect = self.primitive_set[node.rect_idx]
            Wd = rect.Width()
            Ht = rect.Height()

            if node.node_type == NodeType.OrNode:
                # mark symmetric children
                useSplitWds = []
                useSplitHts = []
                if self.param.remove_symmetric_children_of_or_node == 2:
                    child_ids = node.child_ids[::-1]
                else:
                    child_ids = node.child_ids

                for j in child_ids:
                    ch = self.node_set[j]
                    if ch.node_type == NodeType.TerminalNode:
                        continue

                    if ch.split_type == SplitType.VerSplit:
                        if (Wd-ch.split_step2, ch.split_step1) not in useSplitWds:
                            useSplitWds.append((ch.split_step1, Wd-ch.split_step2))
                        else:
                            self.node_set[j].is_symmetric = True

                    elif ch.split_type == SplitType.HorSplit:
                        if (Ht-ch.split_step2, ch.split_step1) not in useSplitHts:
                            useSplitHts.append((ch.split_step1, Ht-ch.split_step2))
                        else:
                            self.node_set[j].is_symmetric = True

    def _find_dbl_counting_or_nodes(self):
        for node in self.node_set:
            if node.node_type != NodeType.OrNode or len(node.child_ids) < 2:
                continue
            for i in self.node_BFS[node.id][1:]:
                npaths = { x : 0 for x in self.node_BFS[node.id] }
                n = self._countPaths(node, self.node_set[i], npaths)
                if n > 1:
                    self.node_set[node.id].has_dbl_counting = True
                    break

    def find_node_ids_with_given_prim_type(self, prim_type, parent_rect=None, node_type=NodeType.TerminalNode):
        ids = []
        for node in self.node_set:
            if node.node_type != node_type:
                continue
            rect = self.primitive_set[node.rect_idx]
            if [rect.Height(), rect.Width()] == prim_type:
                if parent_rect is not None:
                    if rect.x1 >= parent_rect.x1 and rect.y1 >= parent_rect.y1 and \
                            rect.x2 <= parent_rect.x2 and rect.y2 <= parent_rect.y2:
                        ids.append(node.id)
                else:
                    ids.append(node.id)
        return ids

    def Create(self):
        # print("======= creating AOGrid {}, could take a while".format(self.param.tag))
        # FIXME: when remove_symmetric_children_of_or_node is true, top-down hierarchy is not correctly created.

        # the root OrNode
        rect = Rect(0, 0, self.param.grid_wd - 1, self.param.grid_ht - 1)
        self.primitive_set.append(rect)
        node = Node(node_type=NodeType.OrNode, rect_idx=0)
        self._AddNode(node)

        BFS = deque()
        BFS.append(0)
        keepLeft = True
        keepTop = True
        while len(BFS) > 0:
            curId = BFS.popleft()
            curNode = self.node_set[curId]
            curRect = self.primitive_set[curNode.rect_idx]
            curWd = curRect.Width()
            curHt = curRect.Height()

            childIds = []

            if curNode.node_type == NodeType.OrNode:
                num_child_node_kept = 0
                # add a terminal node for a non-root OrNode
                allowTerminate = not ((self.param.not_use_large_terminal_node and
                                      float(curWd * curHt) / float(self.param.grid_ht * self.param.grid_wd) >
                                      self.param.turn_off_size_ratio_terminal_node)  or
                                      (self.param.not_use_intermediate_TerminalNode and (curWd > self.param.min_size or curHt > self.param.min_size)))

                if (curId > 0 and allowTerminate) or (curId==0 and self.param.use_root_terminal_node):
                    node = Node(node_type=NodeType.TerminalNode, rect_idx=curNode.rect_idx)
                    suc, node = self._AddNode(node)
                    childIds.append(node.id)
                    num_child_node_kept += 1

                # add all AndNodes for horizontal and vertical binary splits
                if not self._DoSplit(curRect):
                    childIds = list(set(childIds))
                    self.node_set[curId].child_ids = childIds
                    continue

                num_child_node_to_add = self.param.max_children_kept_for_or - num_child_node_kept
                stepH = self._SplitStep(curWd)
                stepV = self._SplitStep(curHt)
                num_stepH = curHt - stepH + 1 - stepH
                num_stepV = curWd - stepV + 1 - stepV
                if num_stepH == 0 and num_stepV == 0:
                    childIds = list(set(childIds))
                    self.node_set[curId].child_ids = childIds
                    continue

                num_child_node_to_add_H = num_stepH / float(num_stepH + num_stepV) * num_child_node_to_add
                num_child_node_to_add_V = num_child_node_to_add - num_child_node_to_add_H

                stepH_step = int(
                    max(1, floor(float(num_stepH) / num_child_node_to_add_H) if num_child_node_to_add_H > 0 else 0))
                stepV_step = int(
                    max(1, floor(float(num_stepV) / num_child_node_to_add_V) if num_child_node_to_add_V > 0 else 0))

                # horizontal splits
                step = stepH
                num_child_node_added_H = 0

                splitHts = []
                for topHt in range(step, curHt - step + 1, stepH_step):
                    if num_child_node_added_H >= num_child_node_to_add_H:
                        break

                    bottomHt = curHt - topHt
                    if self.param.overlap_ratio > 0:
                        numSplit = int(1 + floor(topHt * self.param.overlap_ratio))
                    else:
                        numSplit = 1
                    for b in range(0, numSplit):
                        splitHts.append((topHt, bottomHt))
                        bottomHt += 1
                        num_child_node_added_H += 1

                if self.param.remove_symmetric_children_of_or_node == 1 and self.param.mark_symmetric_syntatic_subgraph == False:
                    useSplitHts = []
                    for (topHt, bottomHt) in splitHts:
                        if (bottomHt, topHt) not in useSplitHts:
                            useSplitHts.append((topHt, bottomHt))
                elif self.param.remove_symmetric_children_of_or_node == 2 and self.param.mark_symmetric_syntatic_subgraph == False:
                    useSplitHts = []
                    for (topHt, bottomHt) in reversed(splitHts):
                        if (bottomHt, topHt) not in useSplitHts:
                            useSplitHts.append((topHt, bottomHt))
                else:
                    useSplitHts = splitHts

                for (topHt, bottomHt) in useSplitHts:
                    node = Node(node_type=NodeType.AndNode, rect_idx=curNode.rect_idx,
                                split_type=SplitType.HorSplit,
                                split_step1=topHt, split_step2=curHt - bottomHt)
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)

                # vertical splits
                step = stepV
                num_child_node_added_V = 0

                splitWds = []
                for leftWd in range(step, curWd - step + 1, stepV_step):
                    if num_child_node_added_V >= num_child_node_to_add_V:
                        break

                    rightWd = curWd - leftWd
                    if self.param.overlap_ratio > 0:
                        numSplit = int(1 + floor(leftWd * self.param.overlap_ratio))
                    else:
                        numSplit = 1
                    for r in range(0, numSplit):
                        splitWds.append((leftWd, rightWd))
                        rightWd += 1
                        num_child_node_added_V += 1

                if self.param.remove_symmetric_children_of_or_node == 1 and self.param.mark_symmetric_syntatic_subgraph == False:
                    useSplitWds = []
                    for (leftWd, rightWd) in splitWds:
                        if (rightWd, leftWd) not in useSplitWds:
                            useSplitWds.append((leftWd, rightWd))
                elif self.param.remove_symmetric_children_of_or_node == 2 and self.param.mark_symmetric_syntatic_subgraph == False:
                    useSplitWds = []
                    for (leftWd, rightWd) in reversed(splitWds):
                        if (rightWd, leftWd) not in useSplitWds:
                            useSplitWds.append((leftWd, rightWd))
                else:
                    useSplitWds = splitWds

                for (leftWd, rightWd) in useSplitWds:
                    node = Node(node_type=NodeType.AndNode, rect_idx=curNode.rect_idx,
                                split_type=SplitType.VerSplit,
                                split_step1=leftWd, split_step2=curWd - rightWd)
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)

            elif curNode.node_type == NodeType.AndNode:
                # add two child OrNodes
                if curNode.split_type == SplitType.HorSplit:
                    top = Rect(x1=curRect.x1, y1=curRect.y1,
                               x2=curRect.x2, y2=curRect.y1 + curNode.split_step1 - 1)
                    node = Node(node_type=NodeType.OrNode, rect_idx=self._AddPrimitve(top))
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)

                    bottom = Rect(x1=curRect.x1, y1=curRect.y1 + curNode.split_step2,
                                  x2=curRect.x2, y2=curRect.y2)
                    node = Node(node_type=NodeType.OrNode, rect_idx=self._AddPrimitve(bottom))
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)
                elif curNode.split_type == SplitType.VerSplit:
                    left = Rect(curRect.x1, curRect.y1,
                                curRect.x1 + curNode.split_step1 - 1, curRect.y2)
                    node = Node(node_type=NodeType.OrNode, rect_idx=self._AddPrimitve(left))
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)

                    right = Rect(curRect.x1 + curNode.split_step2, curRect.y1,
                                 curRect.x2, curRect.y2)
                    node = Node(node_type=NodeType.OrNode, rect_idx=self._AddPrimitve(right))
                    suc, node = self._AddNode(node)
                    if suc:
                        BFS.append(node.id)
                    childIds.append(node.id)

            childIds = list(set(childIds))
            self.node_set[curId].child_ids = childIds

        # add root terminal node if allowed
        root_id = 0

        # create And-nodes with more than 2 children
        # TODO: handle remove_symmetric_child_node
        if self.param.max_split > 2:
            for branch in range(3, self.param.max_split + 1):
                for node in self.node_set:
                    if node.node_type != NodeType.OrNode:
                        continue

                    new_and_ids = []

                    for cur_id in node.child_ids:
                        cur_and = self.node_set[cur_id]
                        if len(cur_and.child_ids) != branch - 1:
                            continue
                        assert cur_and.node_type == NodeType.AndNode

                        for ch_id in cur_and.child_ids:
                            ch = self.node_set[ch_id]
                            curRect = self.primitive_set[ch.rect_idx]
                            curWd = curRect.Width()
                            curHt = curRect.Height()

                            # split ch into two to create new And-nodes

                            # add all AndNodes for horizontal and vertical binary splits
                            if not self._DoSplit(curRect):
                                continue

                            # horizontal splits
                            step = self._SplitStep(curWd)
                            for topHt in range(step, curHt - step + 1):
                                bottomHt = curHt - topHt
                                if self.param.overlap_ratio > 0:
                                    numSplit = int(1 + floor(topHt * self.param.overlap_ratio))
                                else:
                                    numSplit = 1
                                for b in range(0, numSplit):
                                    split_step1 = topHt
                                    split_step2 = curHt - bottomHt

                                    top = Rect(x1=curRect.x1, y1=curRect.y1,
                                               x2=curRect.x2, y2=curRect.y1 + split_step1 - 1)
                                    top_id = self._FindNodeIdWithGivenRect(top, NodeType.OrNode)
                                    if top_id == -1:
                                        continue
                                    # assert top_id != -1

                                    bottom = Rect(x1=curRect.x1, y1=curRect.y1 + split_step2,
                                                  x2=curRect.x2, y2=curRect.y2)
                                    bottom_id = self._FindNodeIdWithGivenRect(bottom, NodeType.OrNode)
                                    if bottom_id == -1:
                                        continue
                                    # assert bottom_id != -1

                                    # add a new And-node
                                    new_and = Node(node_type=NodeType.AndNode, rect_idx=cur_and.rect_idx)
                                    new_and.child_ids = list(set(cur_and.child_ids) - set([ch_id])) + [top_id,
                                                                                                       bottom_id]

                                    suc, new_and = self._AddNode(new_and)
                                    new_and_ids.append(new_and.id)

                                    bottomHt += 1

                            # vertical splits
                            step = self._SplitStep(curHt)
                            for leftWd in range(step, curWd - step + 1):
                                rightWd = curWd - leftWd

                                if self.param.overlap_ratio > 0:
                                    numSplit = int(1 + floor(leftWd * self.param.overlap_ratio))
                                else:
                                    numSplit = 1
                                for r in range(0, numSplit):
                                    split_step1 = leftWd
                                    split_step2 = curWd - rightWd

                                    left = Rect(curRect.x1, curRect.y1,
                                                curRect.x1 + split_step1 - 1, curRect.y2)
                                    left_id = self._FindNodeIdWithGivenRect(left, NodeType.OrNode)
                                    if left_id == -1:
                                        continue
                                    # assert left_id != -1

                                    right = Rect(curRect.x1 + split_step2, curRect.y1,
                                                 curRect.x2, curRect.y2)
                                    right_id = self._FindNodeIdWithGivenRect(right, NodeType.OrNode)
                                    if right_id == -1:
                                        continue
                                    # assert right_id != -1

                                    # add a new And-node
                                    new_and = Node(node_type=NodeType.AndNode, rect_idx=cur_and.rect_idx)
                                    new_and.child_ids = list(set(cur_and.child_ids) - set([ch_id])) + [left_id,
                                                                                                       right_id]

                                    suc, new_and = self._AddNode(new_and)
                                    new_and_ids.append(new_and.id)

                                    rightWd += 1

                    self.node_set[node.id].child_ids = list(set(self.node_set[node.id].child_ids + new_and_ids))

        self._Postprocessing(root_id)

        # change tnodes to child nodes of and-nodes / or-nodes
        if self.param.use_tnode_as_alpha_channel > 0:
            node_type = NodeType.OrNode if self.param.use_tnode_as_alpha_channel==1 else NodeType.AndNode
            not_create_if_existed = not self.param.use_tnode_as_alpha_channel==1
            for id_ in self.BFS:
                node = self.node_set[id_]
                if node.node_type == NodeType.OrNode and len(node.child_ids) > 1:
                    for ch in node.child_ids:
                        ch_node = self.node_set[ch]
                        if ch_node.node_type == NodeType.TerminalNode:
                            new_parent_node = Node(node_type=node_type, rect_idx=ch_node.rect_idx)
                            _, new_parent_node = self._AddNode(new_parent_node, not_create_if_existed)
                            new_parent_node.child_ids = [ch_node.id, node.id]

                            for pr in node.parent_ids:
                                pr_node = self.node_set[pr]
                                for i, pr_ch in enumerate(pr_node.child_ids):
                                    if pr_ch == node.id:
                                        pr_node.child_ids[i] = new_parent_node.id
                                        break

                            self.node_set[id_].child_ids.remove(ch)
                            if id_ == self.BFS[0]:
                                root_id = new_parent_node.id
                            break

            self._Postprocessing(root_id)

        # add super-or node
        if self.param.use_super_OrNode:
            super_or_node = Node(node_type=NodeType.OrNode, rect_idx=-1)
            _, super_or_node = self._AddNode(super_or_node)
            super_or_node.child_ids = []
            for node in self.node_set:
                if node.node_type == NodeType.OrNode and node.rect_idx != -1:
                    rect = self.primitive_set[node.rect_idx]
                    r = float(rect.Area()) / float(self.param.grid_ht * self.param.grid_wd)
                    if r > 0.5:
                        super_or_node.child_ids.append(node.id)

            root_id = super_or_node.id

            self._Postprocessing(root_id)

        # remove or-nodes with single child node
        if self.param.remove_single_child_or_node:
            remove_ids = []
            for node in self.node_set:
                if node.node_type == NodeType.OrNode and len(node.child_ids) == 1:
                    for pr in node.parent_ids:
                        pr_node = self.node_set[pr]
                        for i, pr_ch in enumerate(pr_node.child_ids):
                            if pr_ch == node.id:
                                pr_node.child_ids[i] = node.child_ids[0]
                                break

                    remove_ids.append(node.id)
                    node.child_ids = []

            remove_ids.sort()
            remove_ids.reverse()

            for id_ in remove_ids:
                for node in self.node_set:
                    if node.id > id_:
                        node.id -= 1
                    for i, ch in enumerate(node.child_ids):
                        if ch > id_:
                            node.child_ids[i] -= 1

                if root_id > id_:
                    root_id -= 1

            for id_ in remove_ids:
                del self.node_set[id_]

            self._Postprocessing(root_id)

        # mark symmetric nodes
        if self.param.mark_symmetric_syntatic_subgraph:
            self._mark_symmetric_subgraph()

        # add tnode hierarchy
        if self.param.use_tnode_topdown_connection:
            self._add_tnode_topdown_connection()
            self._Postprocessing(root_id)
        elif self.param.use_tnode_bottomup_connection:
            self._add_tnode_bottomup_connection()
            self._Postprocessing(root_id)
        elif self.param.use_tnode_bottomup_connection_layerwise:
            self._add_node_bottomup_connection_layerwise()
            self._Postprocessing(root_id)
        elif self.param.use_tnode_bottomup_connection_sequential:
            self._add_tnode_bottomup_connection_sequential()
            self._Postprocessing(root_id)
        elif self.param.use_node_lateral_connection or self.param.use_node_lateral_connection_1:
            root_id = self._add_lateral_connection()
            self._Postprocessing(root_id)

        # index of Or-nodes in BFS
        self.OrNodeIdxInBFS = {}
        self.TNodeIdxInBFS = {}
        idx_or = 0
        idx_t = 0
        for id_ in self.BFS:
            node = self.node_set[id_]
            if node.node_type == NodeType.OrNode:
                self.OrNodeIdxInBFS[node.id] = idx_or
                idx_or += 1
            elif node.node_type == NodeType.TerminalNode:
                self.TNodeIdxInBFS[node.id] = idx_t
                idx_t += 1

        # get DFS and BFS rooted at each node
        for node in self.node_set:
            if node.node_type == NodeType.TerminalNode:
                continue
            visited = np.zeros(len(self.node_set))
            self.node_DFS[node.id] = []
            self.node_DFS[node.id], _ = self._DFS(node.id, self.node_DFS[node.id], visited)

            visited = np.zeros(len(self.node_set))
            self.node_BFS[node.id] = []
            self.node_BFS[node.id], _ = self._BFS(node.id, self.node_BFS[node.id], visited)

        # count paths between nodes and root node
        for n in self.node_set:
            npaths = { x.id : 0 for x in self.node_set }
            self.node_set[n.id].npaths = self._countPaths(self.node_set[self.BFS[0]], n, npaths)

        # find ornode with double-counting children
        self._find_dbl_counting_or_nodes()

        # generate colors for terminal nodes for consistency in visualization
        self.TNodeColors = {}
        for node in self.node_set:
            if node.node_type == NodeType.TerminalNode:
                self.TNodeColors[node.id] = (
                    random.random(), random.random(), random.random())  # generate a random color


    def TurnOnOffNodes(self, on_off):
        for i in range(len(self.node_set)):
            self.node_set[i].on_off = on_off

    def UpdateOnOffNodes(self, pg, offset_using_part_type, class_name=''):
        BFS = [self.BFS[0]]
        pg_used = np.ones((1, len(pg)), dtype=np.int) * -1
        configuration = []
        tnode_offset_indx = []
        while len(BFS):
            id = BFS.pop()
            node = self.node_set[id]
            self.node_set[id].on_off = True
            if len(class_name):
                if class_name in node.which_classes_visited.keys():
                    self.node_set[id].which_classes_visited[class_name] += 1.0
                else:
                    self.node_set[id].which_classes_visited[class_name] = 0

            if node.node_type == NodeType.OrNode:
                idx = self.OrNodeIdxInBFS[node.id]
                BFS.append(node.child_ids[int(pg[idx])])
                pg_used[0, idx] = int(pg[idx])
                if len(self.node_set[id].out_edge_visited_count):
                    self.node_set[id].out_edge_visited_count[int(pg[idx])] += 1.0
                else:
                    self.node_set[id].out_edge_visited_count = np.zeros((len(node.child_ids),), dtype=np.float32)
            elif node.node_type == NodeType.AndNode:
                BFS += node.child_ids

            else:
                configuration.append(node.id)

                offset_ind = 0
                if not offset_using_part_type:
                    for node1 in self.node_set:
                        if node1.node_type == NodeType.TerminalNode:  # change to BFS after _part_instance is changed to BFS
                            if node1.id == node.id:
                                break
                            offset_ind += 1
                else:
                    rect = self.primitive_set[node.rect_idx]
                    offset_ind = self.part_type.index([rect.Height(), rect.Width()])

                tnode_offset_indx.append(offset_ind)

        configuration.sort()
        cfg = np.ones((1, self.num_TNodes), dtype=np.int) * -1
        cfg[0, :len(configuration)] = configuration
        return pg_used, cfg, tnode_offset_indx

    def ResetOutEdgeVisitedCountNodes(self):
        for i in range(len(self.node_set)):
            self.node_set[i].out_edge_visited_count = []

    def NormalizeOutEdgeVisitedCountNodes(self, count=0):
        if count == 0:
            for i in range(len(self.node_set)):
                if len(self.node_set[i].out_edge_visited_count):
                    count = max(count, max(self.node_set[i].out_edge_visited_count))

        if count == 0:
            return

        for i in range(len(self.node_set)):
            if len(self.node_set[i].out_edge_visited_count):
                self.node_set[i].out_edge_visited_count /= count

    def ResetWhichClassesVisitedNodes(self):
        for i in range(len(self.node_set)):
            self.node_set[i].which_classes_visited = {}

    def NormalizeWhichClassesVisitedNodes(self, class_name, count):
        assert count > 0
        for i in range(len(self.node_set)):
            if class_name in self.node_set[i].which_classes_visited.keys():
                self.node_set[i].which_classes_visited[class_name] /= count
