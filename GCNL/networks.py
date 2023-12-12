#!/usr/bin/env python3
# -*- coding: utf-8



import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from logzero import logger

from trunk.deepgraphgo.modules import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
__all__ = ['GcnNet']


class GcnNet(nn.Module):
    """
    """

    def __init__(self, *, labels_num, input_size, hidden_size, num_gcn=0, dropout=0.5, residual=True,
                 **kwargs):
        super(GcnNet, self).__init__()
        logger.info(F'GCN: labels_num={labels_num}, input size={input_size}, hidden_size={hidden_size}, '
                    F'num_gcn={num_gcn}, dropout={dropout}, residual={residual}')
        self.labels_num = labels_num
        self.input_size = input_size


        self.dropout = nn.Dropout(dropout)
        # self.update = nn.ModuleList(NodeUpdate(hidden_size, hidden_size, dropout) for _ in range(num_gcn))
        # self.output = nn.Linear(hidden_size, self.labels_num)  # layers including output
        self.residual = residual
        self.num_gcn = num_gcn  #2 num_hops：要进行的采样层数，即每个子图的邻居层数。 #层数
        self.task_indices = []
        # self.reset_parameters()

        # DEN model 更新
        self.T = 0
        self.batch_size = 64
        self.task_indices = []
        self.dims = [self.input_size, 521, 521, self.labels_num]  # "Dimensions about layers including output"
        self.n_layers = len(self.dims) - 1       # 层数：2层
        self.params = dict()    # 存放DEN各种参数的
        self.GcnParams = dict()  # 存放GCN各种参数的
        self.ex_k = 10  # "The number of units increased in the expansion processing"
        self.param_trained = set()
        self.GcnParams_trained = set()
        self.max_iter = 700  # "Epoch to train"
        self.init_lr = 0.001  # "Learing rate(init) for train"
        self.l1_lambda = 0.00001  # "Sparsity for L1"
        self.l2_lambda = 0.0001  # "L2 lambda"
        self.gl_lambda = 0.001  # "Group Lasso lambda"
        self.regular_lambda = 0.5  # "regularization lambda"
        self.early_training = self.max_iter / 10.
        self.time_stamp = dict()   # 时间戳  记录每一次更新的层数
        self.loss_thr = 0.0001  # "Threshold of dynamic expansion"
        self.spl_thr = 0.005  # "Threshold of split and duplication"

        # self.reset_parameters() # 重置
        # self.input = nn.EmbeddingBag(input_size, hidden_size, mode='sum', include_last_offset=True)
        # self.input_bias = nn.Parameter(torch.zeros(hidden_size))

        self.input = None  # layers including input 输入层
        self.input_bias = None  # 隐藏层


        # self.lstm = nn.LSTM(512,512,num_layers=2)

        for i in range(self.n_layers - 1):   # 定义了2层 w 和 b
            w = self.create_variable('layer%d' % (i + 1), 'weight', [self.dims[i], self.dims[i + 1]])
            b = self.create_variable('layer%d' % (i + 1), 'biases', [self.dims[i + 1]])
            self.params[w.name] = w
            self.params[b.name] = b

        self.cur_W, self.prev_W = dict(), dict()


        # for i in range(self.n_layers - 1):   # 定义了2层 w 和 b
        #     w = self.create_variable('update%d' % (i + 1), 'weight', [self.dims[i], self.dims[i + 1]])
        #     b = self.create_variable('update%d' % (i + 1), 'biases', [self.dims[i + 1]])
        #     self.GcnParams[w.name] = w
        #     self.GcnParams[b.name] = b
        #
        # self.cur_GCN_W, self.prev_GCN_W = dict(), dict()
    # DEN 中定义两层的 w 和 b
    def create_variable(self, scope, name, shape, trainable = True):
        with tf.variable_scope(scope):
            w = tf.get_variable(name, shape, trainable = trainable)
            if 'new' not in w.name:
                self.params[w.name] = w
        return w
        # DEN 中定义两层的 w 和 b

    def create_GCN_variable(self, scope, name, shape, trainable=True):
        with tf.variable_scope(scope):
            w = tf.get_variable(name, shape, trainable=trainable)
            if 'new' not in w.name:
                self.GcnParams[w.name] = w
        return w

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input.weight)
        for update in self.update:
            update.reset_parameters()   #  nn.init.xavier_uniform_(self.ppi_linear.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, nf: dgl.NodeFlow, inputs):   # inputs = 输入稀疏矩阵
        nf.copy_from_parent()
        outputs = self.dropout(F.relu(self.input(*inputs) + self.input_bias))  # 输入的稀疏矩阵分别进行了输入层的输入、ReLU激活以及Dropout操作
        # F.relu() 函数表示对输入进行 ReLU 激活操作，将所有小于零的值都变为零
        # self.input(*inputs) + self.input_bias=计算输入的加权和
        # self.dropout() 函数来对输出进行 Dropout 操作，即随机将一部分输出的值设为零，以防止过拟合

        # outputs, _ = self.lstm(outputs)

        nf.layers[0].data['h'] = outputs      # GCN的第1层保存
        firstlayer = self.state_dict()
        # logger.info(F'First Layer params: {firstlayer}')
        # nf.layers[0] 表示 GCN 模型中的第一个图卷积层         .data['h'] 则是第1层节点的特征向量组成的 Tensor
        # outputs = Tensor，张量 ，多维数组数据结构
        # 前面一层 GCN 模型的输出结果 赋值给 nf.layers[0].data['h']

        # 在 GCN 模型中，每一层的特征都是由前一层的特征和邻居节点的信息共同决定的。
        # 因此，通过这种聚合操作能够有效地利用邻居节点的信息来更新节点特征
        # firstself = nf.layers[0].data['self']
        for i, update in enumerate(self.update):
            if self.residual:  # 残差连接
                nf.block_compute(i,
                                 dgl.function.u_mul_e('h', 'self', out='m_res'),
                                 dgl.function.sum(msg='m_res', out='res'))
                # 第 i 层节点
                # dgl.function.u_mul_e()  将节点特征向量 h 与边权重矩阵 self 进行点乘操作，并将结果赋值给 m_res
                # dgl.function.sum() 则将所有边信息的和汇总到某个目标节点上，并将结果赋值给 res

            firstlayer = self.state_dict()
            # logger.info(F'First Layer params: {firstlayer}')
            nf.block_compute(i,
                             dgl.function.u_mul_e('h', 'ppi', out='ppi_m_out'),  # 节点特征向量与边权重矩阵进行点乘
                             dgl.function.sum(msg='ppi_m_out', out='ppi_out'), update)

            Secondlayer = self.state_dict()
            # logger.info(F'First Layer params: {Secondlayer}')

        return self.output(nf.layers[-1].data['h'])    #返回最后一层的 = scores
        # nf.layers[-1]神经网络中的最后一层
        # nf.layers[-1].data['h'] 则表示最后一层节点的特征向量组成的 Tensor
