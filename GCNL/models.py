#!/usr/bin/env python3
# -*- coding: utf-8


import numpy as np
import torch
import torch.nn as nn
import dgl
from pathlib import Path
from tqdm import tqdm
from logzero import logger

from trunk.deepgraphgo.networks import GcnNet
from trunk.deepgraphgo.evaluation import fmax, aupr
from .modules import NodeUpdate

# DEN dependences
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import re, random, collections
from collections import defaultdict
from numpy import linalg as LA
# from .ops import *
import warnings

warnings.filterwarnings('ignore')


__all__ = ['Model']


class Model(object):
    """

    """

    def __init__(self, *, model_path: Path, dgl_graph, network_x, **kwargs):
        self.model = self.network = GcnNet(**kwargs)
        self.dp_network = nn.DataParallel(self.network)  # 对神经网络模型进行并行化处理
        model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.loss_fn = nn.BCEWithLogitsLoss()
        # self.optimizer = self.model.params  # L
        # self.GCN_optimizer = self.model.GcnParams()  # GCNopt
        self.dgl_graph, self.network_x, self.batch_size = dgl_graph, network_x, None

        # DEN参数
        self.batch_size = 64
        # self.labels_num = self.model.labels_num

    # 获取train的分数矩阵
    def get_scores(self, nf: dgl.NodeFlow):
        batch_x = self.network_x[nf.layer_parent_nid(0).numpy()]
        scores = self.network(nf, (torch.from_numpy(batch_x.indices).long(),
                                   torch.from_numpy(batch_x.indptr).long(),
                                   torch.from_numpy(batch_x.data).float()))  # 到networks ->forword
        return scores  # 返回第二层之后的模型的输出分数，即当前批次训练数据的预测值。

    # 以便进行参数更新, AdamW 优化器是一种改进后的 Adam 优化器
    def get_optimizer(self, **kwargs):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **kwargs)

    #  train_x:当前批次训练数据节点特征的张量或稀疏矩,train_y:数据标签
    #  ,update参数表示是否需要更新模型参数，如果为 True，则进行梯度下降更新
    def train_step(self, train_x, train_y, update, **kwargs):
        self.model.train()  # 训练模式
        scores = self.get_scores(train_x)  # 预测矩阵:scores
        loss = self.loss_fn(scores, train_y)  # nn.BCEWithLogitsLoss()
        # 计算损失:scores（tensor）,当前批次的分数；train_y（tensor）,当前批次的标签值
        loss.backward()  # 反向传播，然后参数更新
        if update:
            self.optimizer.step(closure=None)
            self.optimizer.zero_grad()
        return loss.item()

    @torch.no_grad()
    def predict_step(self, data_x):
        self.model.eval()  # 用于将模型设置为评估模式
        return torch.sigmoid(self.get_scores(data_x)).cpu().numpy()  # 预测的时候获取概率分数矩阵

    # 待预测：预测valid/test的分数矩阵
    def predict(self, test_ppi, batch_size=None, valid=False, **kwargs):
        if batch_size is None:
            batch_size = self.model.batch_size
        # if not valid:
        #     self.load_model()
        unique_test_ppi = np.unique(test_ppi)  # 获取测试集数据中有多少个唯一的节点 ID
        mapping = {x: i for i, x in enumerate(unique_test_ppi)}
        test_ppi = np.asarray([mapping[x] for x in test_ppi])
        scores = np.vstack([self.predict_step(nf)
                            for nf in
                            dgl.contrib.sampling.sampler.NeighborSampler(self.dgl_graph, self.model.batch_size,
                                                                         self.dgl_graph.number_of_nodes(),
                                                                         num_hops=self.model.num_gcn,  # 采样邻居两层
                                                                         seed_nodes=unique_test_ppi,
                                                                         prefetch=True)])
        return scores[test_ppi]

    def save_model(self):  # 保存模型
        torch.save(self.model.state_dict(), self.model_path)
        # self.model.state_dict() 表示获取模型的参数字典  # path: 其中包含模型参数的二进制数据

    def load_model(self):  # 加载模型  # 只是用在了每一个TASK结束后的测试中
        # 可以在一个任务来临之后，加载上一个模型的参数
        self.model.load_state_dict(torch.load(self.model_path))

    # DEN算法 获取GCN的参数
    def get_params(self):
        """ Access the parameters """
        mdict = dict()
        for scope_name, param in self.model.params.items():
            w = self.sess.run(param)
            mdict[scope_name] = w
        return mdict
        # DEN算法 获取GCN的参数

    def get_GCN_params(self):
        """ Access the parameters """
        mdict = dict()
        for scope_name, param in self.model.GcnParams.items():
            w = self.sess.run(param)
            mdict[scope_name] = w
        return mdict

    # DEN算法 加载模型参数
    def load_params(self, params, top=False, time=999):
        """ parmas: it contains weight parameters used in network, like ckpt """
        self.model.params = dict()
        if top:
            # for last layer nodes
            for scope_name, param in params.items():
                scope_name = scope_name.split(':')[0]
                if ('layer%d' % self.model.n_layers in scope_name) and (('_%d' % self.model.T) in scope_name):
                    w = tf.get_variable(scope_name, initializer=param, trainable=True)
                    self.model.params[w.name] = w
                elif 'layer%d' % self.model.n_layers in scope_name:
                    w = tf.get_variable(scope_name, initializer=param, trainable=False)
                    self.model.params[w.name] = w
                else:
                    pass
            return;

        if time == 1:
            self.model.prev_W = dict()
        for scope_name, param in params.items():
            trainable = True
            if time == 1 and 'layer%d' % self.model.n_layers not in scope_name:
                self.model.prev_W[scope_name] = param
            scope_name = scope_name.split(':')[0]
            scope = scope_name.split('/')[0]
            name = scope_name.split('/')[1]
            if (scope == 'layer%d' % self.model.n_layers) and ('_%d' % self.model.T) not in name: trainable = False
            if (scope in self.model.param_trained): trainable = False
            # current task is trainable
            w = tf.get_variable(scope_name, initializer=param, trainable=trainable)
            self.model.params[w.name] = w

    def load_Gcn_params(self, Gcnparams, top=False, time=999):
        """ parmas: it contains weight parameters used in network, like ckpt """
        self.model.GcnParams = dict()
        if top:
            # for last layer nodes
            for scope_name, param in Gcnparams.items():
                scope_name = scope_name.split(':')[0]
                if ('layer%d' % self.model.n_layers in scope_name) and (('_%d' % self.model.T) in scope_name):
                    w = tf.get_variable(scope_name, initializer=param, trainable=True)
                    self.model.GcnParams[w.name] = w
                elif 'layer%d' % self.model.n_layers in scope_name:
                    w = tf.get_variable(scope_name, initializer=param, trainable=False)
                    self.model.GcnParams[w.name] = w
                else:
                    pass
            return;

        if time == 1:
            self.model.prev_GCN_W = dict()
        for scope_name, param in Gcnparams.items():
            trainable = True
            if time == 1 and 'layer%d' % self.model.n_layers not in scope_name:
                self.model.prev_GCN_W[scope_name] = param
            scope_name = scope_name.split(':')[0]
            scope = scope_name.split('/')[0]
            name = scope_name.split('/')[1]
            if (scope == 'layer%d' % self.model.n_layers) and ('_%d' % self.model.T) not in name: trainable = False
            if (scope in self.model.GcnParams_trained): trainable = False
            # current task is trainable
            w = tf.get_variable(scope_name, initializer=param, trainable=trainable)
            self.model.GcnParams[w.name] = w




    def load_Pre_model(self, params):  # 加载之前的一个模型
        self.model.load_state_dict(params)

    def get_variable(self, scope, name, trainable=True):
        with tf.variable_scope(scope, reuse=True):
            w = tf.get_variable(name, trainable=trainable)
            self.model.params[w.name] = w
        return w

    # # 10
    # # 注意力机制，效果会变差
    # def attention(inputs, prev_hidden):
    #     hidden_size = prev_hidden.get_shape().as_list()[-1]
    #     input_size = inputs.get_shape().as_list()[-1]
    #     W1 = tf.Variable(tf.random_normal([input_size, hidden_size], stddev=0.1), name='W1')
    #     W2 = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1), name='W2')
    #     v = tf.Variable(tf.random_normal([hidden_size], stddev=0.1), name='v')
    #     score = tf.matmul(tf.tanh(tf.matmul(inputs, W1) + tf.matmul(prev_hidden, W2)), v)
    #     attention_weights = tf.nn.sigmoid(score, axis=1)
    #     context_vector = tf.reduce_sum(inputs * tf.expand_dims(attention_weights, -1), axis=1)
    #     return context_vector

    def extend_bottom(self, scope, ex_k=10):
        """ bottom layer expansion. scope is range of layer 底层扩展。范围是层的范围"""
        w = self.get_variable(scope, 'weight')
        b = self.get_variable(scope, 'biases')  # 偏置项
        prev_dim = w.get_shape().as_list()[0]
        new_w = self.model.create_variable('new', 'bw', [prev_dim, ex_k])
        new_b = self.model.create_variable('new', 'bb', [ex_k])

        expanded_w = tf.concat([w, new_w], 1)
        expanded_b = tf.concat([b, new_b], 0)

        self.model.params[w.name] = expanded_w
        self.model.params[b.name] = expanded_b
        return expanded_w, expanded_b

    # 10
    def extend_top(self, scope, ex_k=10):
        """ top layer expansion. scope is range of layer 顶层扩展。范围是层的范围"""
        if 'layer%d' % self.model.n_layers == scope:
            # extend for all task layer
            for i in self.model.task_indices:
                if i == self.model.T:
                    w = self.get_variable(scope, 'weight_%d' % i, True)
                    b = self.get_variable(scope, 'biases_%d' % i, True)
                    new_w = tf.get_variable('new/n%d' % i, [ex_k, self.model.labels_num], trainable=True)
                else:
                    w = self.get_variable(scope, 'weight_%d' % i, False)
                    b = self.get_variable(scope, 'biases_%d' % i, False)
                    new_w = tf.get_variable('new/n%d' % i, [ex_k, self.model.labels_num],
                                            initializer=tf.constant_initializer(0.0), trainable=False)

                expanded_w = tf.concat([w, new_w], 0)
                self.model.params[w.name] = expanded_w
                self.model.params[b.name] = b
            return expanded_w, b
        else:
            w = self.get_variable(scope, 'weight')
            b = self.get_variable(scope, 'biases')

            level = int(re.findall(r'layer(\d)', scope)[0])
            expanded_n_units = self.expansion_layer[self.model.n_layers - level - 2]  # top-down

            next_dim = w.get_shape().as_list()[1]
            new_w = tf.get_variable(scope + 'new_tw', [self.model.ex_k, next_dim], trainable=True)

            expanded_w = tf.concat([w, new_w], 0)
            self.model.params[w.name] = expanded_w
            self.model.params[b.name] = b
            return expanded_w, b

    def extend_param(self, scope, ex_k):
        if 'layer%d' % self.model.n_layers == scope:
            for i in self.model.task_indices:
                if i == self.model.T:  # current task(fragile)
                    w = self.get_variable(scope, 'weight_%d' % i, True)
                    b = self.get_variable(scope, 'biases_%d' % i, True)
                    new_w = tf.get_variable('new_fc/n%d' % i, [ex_k, self.model.labels_num], trainable=True)
                else:
                    # previous tasks
                    w = self.get_variable(scope, 'weight_%d' % i, False)
                    b = self.get_variable(scope, 'biases_%d' % i, False)
                    new_w = tf.get_variable('new_fc/n%d' % i, [ex_k, self.model.labels_num],
                                            initializer=tf.constant_initializer(0.0), trainable=False)
                expanded_w = tf.concat([w, new_w], 0)
                self.model.params[w.name] = expanded_w
                self.model.params[b.name] = b
            return expanded_w, b
        else:
            w = self.get_variable(scope, 'weight')
            b = self.get_variable(scope, 'biases')

            prev_dim = w.get_shape().as_list()[0]
            next_dim = w.get_shape().as_list()[1]
            # connect bottom to top
            new_w = self.model.create_variable(scope + '/new_fc', 'bw', [prev_dim, ex_k])
            new_b = self.model.create_variable(scope + '/new_fc', 'bb', [ex_k])

            expanded_w = tf.concat([w, new_w], 1)
            expanded_b = tf.concat([b, new_b], 0)
            # connect top to bottom
            new_w2 = self.model.create_variable(scope + '/new_fc', 'tw', [ex_k, next_dim + ex_k])

            expanded_w = tf.concat([expanded_w, new_w2], 0)
            self.model.params[w.name] = expanded_w
            self.model.params[b.name] = expanded_b
            return expanded_w, expanded_b

    def build_model(self, task_id, prediction=False, splitting=False, expansion=None):
        bottom = self.X
        if splitting:
            for i in range(1, self.model.n_layers):
                prev_w = np.copy(self.prev_W_split['layer%d' % i + '/weight:0'])
                cur_w = np.copy(self.cur_W['layer%d' % i + '/weight:0'])
                indices = self.unit_indices['layer%d' % i]
                next_dim = prev_w.shape[1]
                if i >= 2 and i < self.model.n_layers:
                    below_dim = prev_w.shape[0]
                    below_indices = self.unit_indices['layer%d' % (i - 1)]
                    bottom_p_prev_ary, bottom_p_new_ary, bottom_c_prev_ary, bottom_c_new_ary = [], [], [], []
                    for j in range(below_dim):
                        if j in below_indices:
                            bottom_p_prev_ary.append(prev_w[j, :])
                            bottom_p_new_ary.append(cur_w[j, :])
                            bottom_c_prev_ary.append(cur_w[j, :])
                            bottom_c_new_ary.append(cur_w[j, :])
                        else:
                            bottom_p_prev_ary.append(cur_w[j, :])
                            bottom_c_prev_ary.append(cur_w[j, :])
                    prev_w = np.array(bottom_p_prev_ary + bottom_p_new_ary).astype(np.float32)
                    cur_w = np.array(bottom_c_prev_ary + bottom_c_new_ary).astype(np.float32)

                prev_ary = []
                new_ary = []
                for j in range(next_dim):
                    if j in indices:
                        prev_ary.append(prev_w[:, j])
                        new_ary.append(cur_w[:, j])  # will be expanded
                    else:
                        prev_ary.append(cur_w[:, j])
                # fully connected, L1
                expanded_w = np.array(prev_ary + new_ary).T.astype(np.float32)
                expanded_b = np.concatenate((self.prev_W_split['layer%d' % i + '/biases:0'],
                                             np.random.rand(len(new_ary)))).astype(np.float32)
                with tf.variable_scope('layer%d' % i):
                    w = tf.get_variable('weight', initializer=expanded_w, trainable=True)
                    b = tf.get_variable('biases', initializer=expanded_b, trainable=True)
                self.model.params[w.name] = w
                self.model.params[b.name] = b
                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

                # # # Attention mechanism
                # # #  执行attention 效果会变差，先注释；如果启用，284注释，286-296取消注释，143-154取消注释
                # if i != self.n_layers - 1:
                #     logger.info(F"执行attention!!!!!!!!")
                #     # print("执行attention!!!!!!!!")
                #     attn = tf.layers.dense(bottom, units=next_dim, activation=tf.nn.tanh, name='attention_layer%d' % i)
                #     attn = tf.layers.dense(attn, units=1, activation=None, name='attention_weight%d' % i)
                #     attn = tf.nn.softmax(attn,dim=1)
                #     bottom = tf.matmul(bottom, w) * attn + b
                # else:
                #     bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

            w, b = self.extend_top('layer%d' % self.model.n_layers, len(new_ary))
            self.y = tf.matmul(bottom, w) + b
        elif expansion:
            for i in range(1, self.model.n_layers):
                if i == 1:
                    w, b = self.extend_bottom('layer%d' % i, self.model.ex_k)
                else:
                    w, b = self.extend_param('layer%d' % i, self.model.ex_k)
                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

            w, b = self.extend_param('layer%d' % self.model.n_layers, self.model.ex_k)
            self.y = tf.matmul(bottom, w) + b
        elif prediction:  # 测试
            stamp = self.model.time_stamp['task%d' % task_id]
            for i in range(1, self.model.n_layers):
                w = self.get_variable('layer%d' % i, 'weight', False)
                b = self.get_variable('layer%d' % i, 'biases', False)
                w = w[:stamp[i - 1], :stamp[i]]
                b = b[:stamp[i]]
                logger.info(F' [*] lay %d, shape : %s' % (i, w.get_shape().as_list()))
                # print(' [*] task %d, shape : %s' % (i, w.get_shape().as_list()))

                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)

            w = self.get_variable('layer%d' % self.model.n_layers, 'weight_%d' % task_id, False)
            b = self.get_variable('layer%d' % self.model.n_layers, 'biases_%d' % task_id, False)
            w = w[:stamp[self.model.n_layers - 1], :stamp[self.model.n_layers]]
            b = b[:stamp[self.model.n_layers]]
            self.y = tf.matmul(bottom, w) + b
        else:
            for i in range(1, self.model.n_layers):  # 初始化输出：第一层，第二层
                w = self.get_variable('layer%d' % i, 'weight', True)  # 获取第一层的w
                b = self.get_variable('layer%d' % i, 'biases', True)  # 获取第一层的b
                bottom = tf.nn.relu(tf.matmul(bottom, w) + b)
            prev_dim = bottom.get_shape().as_list()[1]
            w = self.model.create_variable('layer%d' % self.model.n_layers, 'weight_%d' % task_id,
                                           [prev_dim, self.model.labels_num], True)  # 创建第二层的w
            b = self.model.create_variable('layer%d' % self.model.n_layers, 'biases_%d' % task_id,
                                           [self.model.labels_num], True)  # 创建第二层的b
            self.y = tf.matmul(bottom, w) + b  # bottom * w + b   == 第二层的   bottom*w+b

        self.yhat = tf.nn.sigmoid(self.y)  # sigmoid求y^ 的scores

        # loss_all = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(self.y, self.Y)
        # self.loss = tf.reduce_mean(loss_all)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y, labels=self.Y))  # 求loss =−ylog(p)−(1−y)log(1−p)
        if prediction:
            return;

    def selective_learning(self, task_id, selected_params):
        bottom = self.X
        for i in range(1, self.model.n_layers):
            with tf.variable_scope('layer%d' % i):
                w = tf.get_variable('weight', initializer=selected_params['layer%d/weight:0' % i])  # 第一层w
                b = tf.get_variable('biases', initializer=selected_params['layer%d/biases:0' % i])  # 第一层b
            bottom = tf.nn.relu(tf.matmul(bottom, w) + b)
        # last layer
        with tf.variable_scope('layer%d' % self.model.n_layers):
            w = tf.get_variable('weight_%d' % task_id,
                                initializer=selected_params['layer%d/weight_%d:0' % (self.model.n_layers, task_id)])
            b = tf.get_variable('biases_%d' % task_id,
                                initializer=selected_params['layer%d/biases_%d:0' % (self.model.n_layers, task_id)])

        self.y = tf.matmul(bottom, w) + b  # bottom*w + b
        self.yhat = tf.nn.sigmoid(self.y)  # prediction score
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y, labels=self.Y))  # loss

    # 优化更新
    def optimization(self, prev_W, selective=False, splitting=False, expansion=None):
        if selective:
            all_var = [var for var in tf.trainable_variables() if 'layer%d' % self.model.n_layers in var.name]
        else:
            all_var = [var for var in tf.trainable_variables()]

        l2_losses = []
        for var in all_var:
            l2_losses.append(tf.nn.l2_loss(var))

        opt = tf.train.AdamOptimizer(self.model.lr)
        regular_terms = []

        if not splitting and expansion == None:
            for var in all_var:
                if var.name in prev_W.keys():
                    prev_w = prev_W[var.name]
                    regular_terms.append(tf.nn.l2_loss(var - prev_w))
        else:
            for var in all_var:
                if var.name in prev_W.keys():
                    prev_w = prev_W[var.name]
                    if len(prev_w.shape) == 1:
                        sliced = var[:prev_w.shape[0]]
                    else:
                        sliced = var[:prev_w.shape[0], :prev_w.shape[1]]
                    regular_terms.append(tf.nn.l2_loss(sliced - prev_w))

        losses = self.loss + self.model.l2_lambda * tf.reduce_sum(l2_losses) + \
                 self.model.regular_lambda * tf.reduce_sum(regular_terms)

        opt = tf.train.AdamOptimizer(self.model.lr)  # 优化后的lr？？？ 为模型进行优化的过程中指定了学习率self.model.lr。
        grads = opt.compute_gradients(losses, all_var)
        apply_grads = opt.apply_gradients(grads, global_step=self.model.g_step)

        l1_var = [var for var in tf.trainable_variables()]
        l1_op_list = []
        with tf.control_dependencies([apply_grads]):
            for var in l1_var:
                th_t = tf.fill(tf.shape(var), tf.convert_to_tensor(self.model.l1_lambda))
                zero_t = tf.zeros(tf.shape(var))
                var_temp = var - (th_t * tf.sign(var))
                l1_op = var.assign(tf.where(tf.less(tf.abs(var), th_t), zero_t, var_temp))
                l1_op_list.append(l1_op)

        GL_var = [var for var in tf.trainable_variables() if
                  'new' in var.name and ('bw' in var.name or 'tw' in var.name)]
        gl_op_list = []
        with tf.control_dependencies([apply_grads]):
            for var in GL_var:
                g_sum = tf.sqrt(tf.reduce_sum(tf.square(var), 0))
                th_t = self.model.gl_lambda
                gw = []
                for i in range(var.get_shape()[1]):
                    temp_gw = var[:, i] - (th_t * var[:, i] / g_sum[i])
                    gw_gl = tf.where(tf.less(g_sum[i], th_t), tf.zeros(tf.shape(var[:, i])), temp_gw)
                    gw.append(gw_gl)
                gl_op = var.assign(tf.stack(gw, 1))
                gl_op_list.append(gl_op)

        with tf.control_dependencies(l1_op_list + gl_op_list):
            self.opt = tf.no_op()  # self.opt = tf.no_op()

    # # 传入数据大小，初始化模型参数：self.X =dim[0], self.Y = self.labels_num
    def set_initial_states(self, decay_step):
        self.model.g_step = tf.Variable(0., trainable=False)
        self.model.lr = tf.train.exponential_decay(
            self.model.init_lr,  # Base learning rate.
            self.model.g_step * self.model.batch_size,  # Current index into the dataset.
            decay_step,  # Decay step.
            0.95,  # Decay rate.
            staircase=True)
        self.X = tf.placeholder(tf.float32, [None, self.model.dims[0]])  # 占位符  41311   521
        self.Y = tf.placeholder(tf.float32, [None, self.model.labels_num])  # 占位符  NONE    6640

    def add_task(self, task_id, data, best_fmax, **kwargs):
        trainX, trainY, self.valX, self.valY = data
        self.train_range = np.array(range(trainY.shape[0]))
        data_size = trainX.shape[0]
        self.set_initial_states(data_size)  # self.X =dim[0], self.Y = self.labels_num  定义 输入层，输出层
        self.model.input = nn.EmbeddingBag(self.model.dims[0], self.model.dims[1], mode='sum', include_last_offset=True)  # dim 0 * dim 1
        self.model.input_bias = nn.Parameter(torch.zeros(self.model.dims[1]))
        self.model.output = nn.Linear(self.model.dims[2], self.model.labels_num)  # layers including output   # dim 2 * dim 3
        self.model.update = nn.ModuleList(NodeUpdate(self.model.dims[1], self.model.dims[2], self.model.dropout) for _ in range(self.model.num_gcn))

        expansion_layer = []  # to split
        self.expansion_layer = [0, 0]  # new units
        self.build_model(task_id)  # 建立神经模型，定义网络一二层的w，b

        if self.model.T == 1:
            self.optimization(self.model.prev_W)  # 更新W
            self.sess.run(tf.global_variables_initializer())  # 初始化全局变量
            best_fmax = 0.0

            epochs_num, train_loss_mean, best_fmax, t_, aupr_ = self.run_epoch(self.opt, self.loss, (trainX, trainY),
                                                                               (self.valX, self.valY), best_fmax,
                                                                               'Train', selective=True)
            expansion_layer = [0, 0]
        else:
            # 加载上一个TASK的模型参数
            # 在main加载了上一个TASK的参数模型

            """ SELECTIVE LEARN """
            print(' [*] Selective retraining')
            self.optimization(self.model.prev_W, selective=True)
            self.sess.run(tf.global_variables_initializer())

            epochs_num, train_loss_mean, best_fmax, t_, aupr_ = self.run_epoch(self.opt, self.loss, (trainX, trainY),
                                                                               (self.valX, self.valY), best_fmax,
                                                                               'Train', selective=True)

            params = self.get_params()

            currentParams_GCN = self.model.state_dict()  # 加载上一个TASK的模型参数训练后的参数

            self.destroy_graph()
            self.sess = tf.Session()

            # select the units
            # print(' [*] select the units')
            logger.info(F'select the units')
            selected_prev_params = dict()
            selected_params = dict()
            all_indices = defaultdict(list)  # nonzero unis
            for i in range(self.model.n_layers, 0, -1):
                if i == self.model.n_layers:
                    w = params['layer%d/weight_%d:0' % (i, task_id)]
                    b = params['layer%d/biases_%d:0' % (i, task_id)]
                    for j in range(w.shape[0]):
                        if w[j, 0] != 0:
                            all_indices['layer%d' % i].append(j)
                    selected_params['layer%d/weight_%d:0' % (i, task_id)] = w[np.ix_(all_indices['layer%d' % i], [0])]
                    selected_params['layer%d/biases_%d:0' % (i, task_id)] = b
                else:
                    w = params['layer%d/weight:0' % i]
                    b = params['layer%d/biases:0' % i]
                    top_indices = all_indices['layer%d' % (i + 1)]
                    for j in range(w.shape[0]):
                        if np.count_nonzero(w[j, top_indices]) != 0 or i == 1:
                            all_indices['layer%d' % i].append(j)

                    sub_weight = w[np.ix_(all_indices['layer%d' % i], top_indices)]
                    sub_biases = b[all_indices['layer%d' % (i + 1)]]
                    selected_params['layer%d/weight:0' % i] = sub_weight
                    selected_params['layer%d/biases:0' % i] = sub_biases
                    selected_prev_params['layer%d/weight:0' % i] = \
                        self.model.prev_W['layer%d/weight:0' % i][np.ix_(all_indices['layer%d' % i], top_indices)]
                    selected_prev_params['layer%d/biases:0' % i] = \
                        self.model.prev_W['layer%d/biases:0' % i][all_indices['layer%d' % (i + 1)]]

            # learn only selected params
            logger.info(F'learn only selected params')
            self.set_initial_states(data_size)
            self.selective_learning(task_id, selected_params)
            self.optimization(selected_prev_params)  # 优化选择后的params
            self.sess.run(tf.global_variables_initializer())

            epochs_num, train_loss_mean, best_fmax, t_, aupr_ = self.run_epoch(self.opt, self.loss, (trainX, trainY),
                                                                               (self.valX, self.valY), best_fmax,
                                                                               'Train', selective=True)


            _vars = [(var.name, self.sess.run(var)) for var in tf.trainable_variables() if 'layer' in var.name]

            for item in _vars:
                key, values = item
                selected_params[key] = values

            # union
            for i in range(self.model.n_layers, 0, -1):
                if i == self.model.n_layers:
                    temp_weight = params['layer%d/weight_%d:0' % (i, task_id)]
                    temp_weight[np.ix_(all_indices['layer%d' % i], [0])] = \
                        selected_params['layer%d/weight_%d:0' % (i, task_id)]
                    params['layer%d/weight_%d:0' % (i, task_id)] = temp_weight
                    params['layer%d/biases_%d:0' % (i, task_id)] = selected_params['layer%d/biases_%d:0' % (i, task_id)]
                else:
                    temp_weight = params['layer%d/weight:0' % i]
                    temp_biases = params['layer%d/biases:0' % i]
                    temp_weight[np.ix_(all_indices['layer%d' % i], all_indices['layer%d' % (i + 1)])] = \
                        selected_params['layer%d/weight:0' % i]
                    temp_biases[all_indices['layer%d' % (i + 1)]] = selected_params['layer%d/biases:0' % i]
                    params['layer%d/weight:0' % i] = temp_weight
                    params['layer%d/biases:0' % i] = temp_biases

            """ Network Expansion """
            if train_loss_mean < self.model.loss_thr:
                pass
            else:
                # addition
                self.destroy_graph()
                self.sess = tf.Session()
                self.load_params(params)

                self.load_Pre_model(currentParams_GCN)  # 加载selected params之后的的model的params

                self.set_initial_states(data_size)
                self.build_model(task_id, expansion=True)
                self.optimization(self.model.prev_W, expansion=True)
                self.sess.run(tf.global_variables_initializer())

                print(' [*] Network expansion (training)')

                epochs_num, train_loss_mean, best_fmax, t_, aupr_ = self.run_epoch(self.opt, self.loss,
                                                                                   (trainX, trainY),
                                                                                   (self.valX, self.valY), best_fmax,
                                                                                   'Train', selective=True)

                params = self.get_params()

                currentParams_GCN = self.model.state_dict()  # 加载 Network expansion 模型参数训练后的参数

                for i in range(self.model.n_layers - 1, 0, -1):
                    prev_layer_weight = params['layer%d/weight:0' % i]
                    prev_layer_biases = params['layer%d/biases:0' % i]
                    useless = []
                    for j in range(prev_layer_weight.shape[1] - self.model.ex_k, prev_layer_weight.shape[1]):
                        if np.count_nonzero(prev_layer_weight[:, j]) == 0:
                            useless.append(j)
                    cur_layer_weight = np.delete(prev_layer_weight, useless, axis=1)
                    cur_layer_biases = np.delete(prev_layer_biases, useless)
                    params['layer%d/weight:0' % i] = cur_layer_weight
                    params['layer%d/biases:0' % i] = cur_layer_biases

                    if i == self.model.n_layers - 1:
                        for t in self.model.task_indices:
                            prev_layer_weight = params['layer%d/weight_%d:0' % (i + 1, t)]
                            cur_layer_weight = np.delete(prev_layer_weight, useless, axis=0)
                            params['layer%d/weight_%d:0' % (i + 1, t)] = cur_layer_weight
                    else:
                        prev_layer_weight = params['layer%d/weight:0' % (i + 1)]
                        cur_layer_weight = np.delete(prev_layer_weight, useless, axis=0)
                        params['layer%d/weight:0' % (i + 1)] = cur_layer_weight

                    self.expansion_layer[i - 1] = self.model.ex_k - len(useless)

                    print("   [*] Expanding %dth hidden unit, %d unit added, (valid, repeated: %d)" \
                          % (i, self.expansion_layer[i - 1], epochs_num))

                print(' [*] Split & Duplication')
                self.cur_W = params
                # find the highly drifted ones and split
                self.unit_indices = dict()
                for i in range(1, self.model.n_layers):
                    prev = self.model.prev_W['layer%d/weight:0' % i]
                    cur = params['layer%d/weight:0' % i]
                    next_dim = prev.shape[1]

                    indices = []
                    cosims = []
                    for j in range(next_dim):
                        cosim = LA.norm(prev[:, j] - cur[:prev.shape[0], j])

                        if cosim > self.model.spl_thr:
                            indices.append(j)
                            cosims.append(cosim)
                    _temp = np.argsort(cosims)[:self.model.ex_k]
                    print("   [*] split N in layer%d: %d / %d" % (i, len(_temp), len(cosims)))
                    indices = np.array(indices)[_temp]
                    self.expansion_layer[i - 1] += len(indices)
                    expansion_layer.append(len(indices))
                    self.unit_indices['layer%d' % i] = indices

                self.prev_W_split = self.cur_W.copy()
                for key, values in self.model.prev_W.items():
                    temp = self.prev_W_split[key]
                    if len(values.shape) >= 2:
                        temp[:values.shape[0], :values.shape[1]] = values
                    else:
                        temp[:values.shape[0]] = values
                    self.prev_W_split[key] = temp

                self.destroy_graph()
                self.sess = tf.Session()
                self.load_params(params, top=True)

                self.load_Pre_model(currentParams_GCN)  # 加载 Split & Duplication 之后的的model的params

                self.set_initial_states(data_size)
                self.build_model(task_id, splitting=True)
                self.optimization(self.model.prev_W, splitting=True)
                self.sess.run(tf.global_variables_initializer())

                epochs_num, train_loss_mean, best_fmax, t_, aupr_ = self.run_epoch(self.opt, self.loss,
                                                                                   (trainX, trainY),
                                                                                   (self.valX, self.valY), best_fmax,
                                                                                   'Train', selective=True)

        print("[*] Total expansions: %s" % self.expansion_layer)

        params = self.get_params()

        currentParams_GCN = self.model.state_dict()  # 获取最后的模型参数

        # time stamp
        stamp = []
        for i in range(1, self.model.n_layers + 1):
            if i == self.model.n_layers:
                dim = params['layer%d/weight_%d:0' % (i, task_id)].shape[0]
            else:
                dim = params['layer%d/weight:0' % i].shape[0]
            stamp.append(dim)

        stamp.append(self.model.labels_num)
        self.model.time_stamp['task%d' % task_id] = stamp

        self.destroy_graph()
        self.sess = tf.Session()
        self.load_params(params)
        self.load_Pre_model(currentParams_GCN)  # 加载最后的model的params

        self.set_initial_states(data_size)
        # # 建立模型进行测试
        self.build_model(task_id, prediction=True)
        self.sess.run(tf.global_variables_initializer())
        self.model.param_trained.add('layer1')
        self.model.param_trained.add('layer2')
        return train_loss_mean, best_fmax, aupr_
    def init_model(self):
        self.model.input = nn.EmbeddingBag(self.model.dims[0], self.model.dims[1], mode='sum',
                                           include_last_offset=True)  # dim 0 * dim 1
        self.model.input_bias = nn.Parameter(torch.zeros(self.model.dims[1]))
        self.model.output = nn.Linear(self.model.dims[2],
                                      self.model.labels_num)  # layers including output   # dim 2 * dim 3
        self.model.update = nn.ModuleList(
            NodeUpdate(self.model.dims[1], self.model.dims[2], self.model.dropout) for _ in range(self.model.num_gcn))

    def run_epoch(self, opt, loss, train_data, valid_data, best_fmax, desc='Train', selective=False, print_pred=True):
        opt_params = ()
        currentParams = self.model.parameters()
        self.get_optimizer(**dict(opt_params))
        window_size = 10
        epochs_num = 3
        loss_window = collections.deque(maxlen=window_size)
        # self.get_optimizer(self.optimization) # 没次更新新参数
        batch_size = self.model.batch_size
        (train_ppi, train_y), (valid_ppi, valid_y) = train_data, valid_data
        ppi_train_idx = np.full(self.network_x.shape[0], -1, dtype=np.int)
        ppi_train_idx[train_ppi] = np.arange(train_ppi.shape[0])
        # train_step 获得scores  需要修改
        for epoch_idx in range(epochs_num):
            train_loss_sum = 0.0
            for nf in tqdm(dgl.contrib.sampling.sampler.NeighborSampler(self.dgl_graph, batch_size,
                                                                        self.dgl_graph.number_of_nodes(),
                                                                        neighbor_type='in',
                                                                        num_workers=32,
                                                                        num_hops=self.model.num_gcn,
                                                                        seed_nodes=train_ppi,
                                                                        prefetch=True, shuffle=True),
                           desc=F'Epoch {epoch_idx}', leave=False, dynamic_ncols=True,
                           total=(len(train_ppi) + batch_size - 1) // batch_size):
                batch_y = train_y[ppi_train_idx[nf.layer_parent_nid(-1).numpy()]].toarray()
                # train_loss_single = self.train_step(nf, torch.from_numpy(batch_y), True)
                # fmax_epoch = self.valid(valid_ppi, valid_y, epoch_idx, train_loss_single, best_fmax)
                train_loss_sum += self.train_step(nf, torch.from_numpy(batch_y), True)  # 训练损失（training loss）和模型参数更新
                # run_epoch 训练损失（training loss）
                # 行训练，其中 train_y 是标签矩阵，用于训练分类器。每个 batch 的大小由 batch_size 定义，
                # num_hops 参数定义了需要采样的邻居节点数量，
                # 而 num_workers 控制着要启动多少进程来处理采样任务。该方法可以有效地减小运算时间和内存占用。
                # 通过循环迭代每个 batch 实现了整个数据集的训练过程，并在每个 epoch 结束时累加 train_loss。
                train_loss_mean = train_loss_sum / len(train_ppi)
            best_fmax, t_, aupr_ = self.valid(valid_ppi, valid_y, epoch_idx, train_loss_sum / len(train_ppi), best_fmax)

        return epochs_num, train_loss_mean, best_fmax, t_, aupr_

    # 验证每一个epoch，train后的f，
    def valid(self, valid_loader, targets, epoch_idx, train_loss, best_fmax):
        scores = self.predict(valid_loader, valid=True)
        (fmax_, t_), aupr_ = fmax(targets, scores), aupr(targets.toarray().flatten(), scores.flatten())
        # 返回最优 F1 值以及对应的阈值
        # logger.info(F" [*] iter: %d, F: %.4f, Aupr: %.4f" % (c_iter, val_loss, val_perf, F, Aupr))
        logger.info(F'TASK_TRAIN  Epoch: {epoch_idx}: Loss: {train_loss:.5f} '
                    F'Fmax: {fmax_:.4f} {t_:.3f} AUPR: {aupr_:.4f}')
        if fmax_ > best_fmax:
            best_fmax = fmax_
            self.save_model()  #
        return best_fmax, t_, aupr_


    def destroy_graph(self):
        tf.reset_default_graph()
