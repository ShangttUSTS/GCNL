#!/usr/bin/env python3
# -*- coding: utf-8


import warnings
import click
import numpy as np
import scipy.sparse as ssp
import torch
import dgl.data
from pathlib import Path
from ruamel.yaml import YAML
from logzero import logger

from GCNL.data_utils import get_pid_list, get_data, get_mlb, output_res, get_ppi_idx, get_homo_ppi_idx
from GCNL.models import Model  # LLL初始化在Model中的model中。

# DEN dependences
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import warnings
warnings.filterwarnings('ignore')



@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of model configure yaml.')
@click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
@click.option('--model-id', type=click.INT, default=None)
def main(data_cnf, model_cnf, mode, model_id):
    model_id = F'-Model-{model_id}' if model_id is not None else ''
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    data_name, model_name = data_cnf['name'], model_cnf['name']  # data_name:'mf',model_name:'GCNL_GO'

    model = None  # 初始化model=none
    data_cnf['mlb'] = Path(data_cnf['mlb'])
    data_cnf['results'] = Path(data_cnf['results'])


    # net_pid_list:蛋白质id；['Q54UL4','Q8NA31'] 共189065
    # net_pid_map:蛋白质id编号  ['Q54UL4':0,'Q8NA31':1] 共189065
    net_pid_list = get_pid_list(data_cnf['network']['pid_list'])
    net_pid_map = {pid: i for i, pid in enumerate(net_pid_list)}
    net_blastdb = data_cnf['network']['blastdb']   # data/pip_blastbd
    dgl_graph = dgl.data.utils.load_graphs(data_cnf['network']['dgl'])[0][0]
    self_loop = torch.zeros_like(dgl_graph.edata['ppi'])
    self_loop[dgl_graph.edge_ids(nr_:=np.arange(dgl_graph.number_of_nodes()), nr_)] = 1.0
    dgl_graph.edata['self'] = self_loop
    dgl_graph.edata['ppi'] = dgl_graph.edata['ppi'].float()
    dgl_graph.edata['self'] = dgl_graph.edata['self'].float()
    logger.info(F'{dgl_graph}')
    network_x = ssp.load_npz(data_cnf['network']['feature'])

    if mode is None or mode == 'train':

        train_pid_list, _, train_go = get_data(**data_cnf['train'])
        # tarin_pssm_vector = get_ppi_pssm(_, net_blastdb)
        train_seq = _

        valid_pid_list, _, valid_go = get_data(**data_cnf['valid'])
        valid_seq = _
        mlb = get_mlb(data_cnf['mlb'], train_go)
        labels_num = len(mlb.classes_)

        #LLL终生学习
        # modelL = ModelLLL((train_pid_list, train_seq, train_go,), (valid_pid_list,valid_seq, valid_go), labels_num)



        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # mlb.transform() 方法将输入的 GO 标签列表转换为一个多标签矩阵，
            # 其中每一行对应一个样本，每一列对应一个 GO 类型。若第 i 个样本具有 j 个 GO 类型，
            # 则该矩阵的第 i 行第 j 列的值为 1，否则为 0。
            # 由于多标签矩阵通常是稀疏的，因此使用 astype(np.float32) 进行类型转换，以便更高效地存储和计算
            train_y, valid_y = mlb.transform(train_go).astype(np.float32), mlb.transform(valid_go).astype(np.float32)

        # 筛选pid_list在net_pid_map中存在的蛋白质，减小数据量
        # pid_list索引  pid_listID，  net_pid_map索引  对应train_y标签矩阵
        *_, train_ppi, train_y = get_ppi_idx(train_pid_list, train_y, net_pid_map)

        # 筛选pid_list在net_pid_map中存在的蛋白质（如果不存在找到同源），减小数据量
        # pid_list索引  pid_listID  net_pid_map索引  对应train_y标签矩阵
        *_, valid_ppi, valid_y = get_homo_ppi_idx(valid_pid_list, data_cnf['valid']['fasta_file'],
                                                  valid_y, net_pid_map, net_blastdb,
                                                  data_cnf['results']/F'{data_name}-valid-ppi-blast-out')
        num_task = 2  # 终生学习共训练6个TASK
        task_permutation_train, task_permutation_valid = [], []
        for task in range(num_task):
            task_permutation_train.append(
                np.random.permutation(train_ppi.shape[0]))  # (6, labels_num), 6行，1乘labels_num的向量
            task_permutation_valid.append(np.random.permutation(valid_ppi.shape[0]))
        train_ppiXs, valid_ppiXs, testXs = [], [], []
        train_ppi_lableXs, valid_ppi_lableXs, testXs = [], [], []
        for task in range(num_task):
            train_ppiXs.append(train_ppi[task_permutation_train[task]])
            valid_ppiXs.append(valid_ppi[task_permutation_valid[task]])
            train_ppi_lableXs.append(train_y[task_permutation_train[task], :])
            valid_ppi_lableXs.append(valid_y[task_permutation_valid[task], :])
            # testXs.append(testX[:, task_permutation[task]])
        # 终生学习六个任务，分别输入train,valid 随机的六组对应数据
        # 说明没进一个任务，数据是打乱的。

        logger.info(F'Number of Labels: {labels_num}')
        logger.info(F'Size of Training Set: {len(train_ppi)}')
        logger.info(F'Size of Validation Set: {len(valid_ppi)}')

        # run_name = F'{model_name}{model_id}-{data_name}-Task-{t + 1}'  # run_name:'GCNL_GO-Model_1-mf-Task-1'
        run_name = F'{model_name}{model_id}-{data_name}'  # run_name:'GCNL_GO-Model_1-mf-Task-1'
        model_cnf['model']['model_path'] = Path(data_cnf['model_path']) / F'{run_name}'  # # run_name:'GCNL_GO-Model_1-mf-Task-1'

        model = Model(labels_num=labels_num, dgl_graph=dgl_graph, network_x=network_x,
                      input_size=network_x.shape[1], **model_cnf['model'])

        params = dict() #记录每次训练的参数
        currentParams_GCN = dict() #GCN 的每次参数
        best_fmax = 0.0  #记录每一个TASK得出的F
        # 尝试结合DEN
        for t in range(num_task):
            # 训练和验证:
            run_name = F'{model_name}{model_id}-{data_name}-Task-{t + 1}'
            model_cnf['model']['model_path'] = Path(data_cnf['model_path']) / F'{run_name}'
            model.model_path = Path(data_cnf['model_path']) / F'{run_name}'
            logger.info(F'Model: {model_name}, Path: {model_cnf["model"]["model_path"]}, Dataset: {data_name}')

            data = (train_ppiXs[t], train_ppi_lableXs[t], valid_ppiXs[t], valid_ppi_lableXs[t])

            model.sess = tf.Session()
            logger.info(F"\tLifeLong Learning : TASK %d TRAINING\n" % (t + 1))
            model.model.T = model.model.T + 1
            model.model.task_indices.append(t + 1)
            model.load_params(params, time=1)  # 1.time =1 设置当前模型params为空
            model.load_Pre_model(currentParams_GCN)
            train_loss_mean, best_fmax, aupr_ = model.add_task(t + 1, data, best_fmax, **model_cnf['train'])  # 2. 模型参数已经更新好
            # 将GCN的模型参数保存
            # 当前模型参数
            currentParams_GCN = model.model.state_dict()
            # currentParams = model.model.parameters()


            params = model.get_params()    # 3.得到刚保存的L模型参数
            model.destroy_graph()
            model.sess.close()

            #测试: 完成一个Task train 就测试一次，生成了1个txt
            model.sess = tf.Session()
            model.load_params(params)
            model.load_Pre_model(currentParams_GCN)  # 用之前最优一个模型结果测试
            logger.info(F'\tLifeLong Learning TASK %d : OVERALL EVALUATION' % (t + 1))
            mlb = get_mlb(data_cnf['mlb'])
            labels_num = len(mlb.classes_)
            if model is None:
                model = Model(labels_num=labels_num, dgl_graph=dgl_graph, network_x=network_x,
                              input_size=network_x.shape[1], **model_cnf['model'])
            test_cnf = data_cnf['test']
            test_name = 'test'
            test_pid_list, _, test_go = get_data(**test_cnf)
            test_res_idx_, test_pid_list_, test_ppi, _ = get_homo_ppi_idx(test_pid_list, test_cnf['fasta_file'],
                                                                          None, net_pid_map, net_blastdb,
                                                                          data_cnf['results'] / F'{data_name}-{test_name}'
                                                                                           F'-ppi-blast-out')
            scores = np.zeros((len(test_pid_list), len(mlb.classes_)))
            scores[test_res_idx_] = model.predict(test_ppi, **model_cnf['test'])
            res_path = data_cnf['results'] / F'{run_name}-{test_name}'
            output_res(res_path.with_suffix('.txt'), test_pid_list, mlb.classes_, scores)
            # 使用output_res()函数将得分结果输出到res_path指定的.txt文件中。
            # 这个函数的作用是将分类器对于
            # 测试集中每个样本的预测结果（test_pid_list）、类别标签（mlb.classes_）和预测概率（scores）写入到一个txt文件中
            np.save(res_path, scores)
            logger.info(F'TASK : { t + 1} Scores Test Path : {res_path}')
            model.destroy_graph()
            model.sess.close()
            # 将得分（scores）保存到res_path指定的.npy文件中，该文件可以随时被读取，加载到numpy数组


            # model.train((train_ppi, train_y), (valid_ppi, valid_y), **model_cnf['train'])





if __name__ == '__main__':
    main()
