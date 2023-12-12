#!/usr/bin/env python3
# -*- coding: utf-8


import torch.nn as nn
import torch.nn.functional as F

__all__ = ['NodeUpdate']


class NodeUpdate(nn.Module):
    """

    """
    def __init__(self, in_f, out_f, dropout):
        super(NodeUpdate, self).__init__()
        self.ppi_linear = nn.Linear(in_f, out_f)   #  nn.Linear() 函数定义了一个线性映射 y=x * w +b  最后一层名字为self.ppi_linear
        self.dropout = dropout

    def forward(self, node):
        outputs = self.dropout(F.relu(self.ppi_linear(node.data['ppi_out'])))
        #F.linear(input, self.weight, self.bias)
        # node.data['ppi_out'] 表示当前层所有节点的 PPI（protein-protein interaction）信息，即通过蛋白质相互作用网络获取的邻居节点特征。
        # self.ppi_linear() 则表示通过一个线性变换对这些节点特征进行转换，以更好地利用这些邻居节点信息。
        if 'res' in node.data:
            outputs = outputs + node.data['res']
        return {'h': outputs}

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ppi_linear.weight)
