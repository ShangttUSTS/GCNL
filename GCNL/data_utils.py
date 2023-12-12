#!/usr/bin/env python3


import joblib
import numpy as np
import scipy.sparse as ssp
from pathlib import Path
from collections import defaultdict
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer

from trunk.deepgraphgo.psiblast_utils import blast

__all__ = ['get_pid_list', 'get_go_list', 'get_pid_go', 'get_pid_go_sc', 'get_data', 'output_res', 'get_mlb',
           'get_pid_go_mat', 'get_pid_go_sc_mat', 'get_ppi_idx', 'get_homo_ppi_idx']

# 获取蛋白质ID列表
def get_pid_list(pid_list_file):
    try:
        with open(pid_list_file) as fp:
            return [line.split()[0] for line in fp]
    except TypeError:
        return pid_list_file

 # 获取每一个pid对应的GO条目
def get_go_list(pid_go_file, pid_list):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                pid_go[(line_list:=line.split())[0]].append(line_list[1])
        return [pid_go[pid_] for pid_ in pid_list]
    else:
        return None

 # evaluation
def get_pid_go(pid_go_file):
    if pid_go_file is not None:
        pid_go = defaultdict(list)
        with open(pid_go_file) as fp:
            for line in fp:
                pid_go[(line_list:=line.split('\t'))[0]].append(line_list[1])
        return dict(pid_go)
    else:
        return None

 # evaluation 获取预测的DeepGraphGO-Ensemble-mf-test.txt  id：GO+分数
def get_pid_go_sc(pid_go_sc_file):
    pid_go_sc = defaultdict(dict)
    with open(pid_go_sc_file) as fp:
        for line in fp:
            pid_go_sc[line_list[0]][line_list[1]] = float((line_list:=line.split('\t'))[2])
    return dict(pid_go_sc)

    # SeqIO.parse() 函数读取 FASTA 文件，并将每个序列的 ID 存储在 pid_list 中，
    # 将每个序列的字符串存储在 data_x 中。如果 feature_type 不为空，
    # 则会根据 **kwargs 参数中给出的文件路径载入对应的特征文件（只支持 .npy 和 .npz 格式），
    # 并将 data_x 更新为特征矩阵。
def get_data(fasta_file, pid_go_file=None, feature_type=None, **kwargs):
    pid_list, data_x = [], []
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        pid_list.append(seq.id)
        data_x.append(str(seq.seq))
    if feature_type is not None:
        feature_path = Path(kwargs[feature_type])
        if feature_path.suffix == '.npy':
            data_x = np.load(feature_path)
        elif feature_path.suffix == '.npz':
            data_x = ssp.load_npz(feature_path)
        else:
            raise ValueError(F'Only support suffix of .npy for np.ndarray or .npz for scipy.csr_matrix as feature.')
    # 每个序列的 ID 存储在 pid_list, 将每个序列的字符串存储在 data_x 中，get_go_list或者pid_list的每一个go条目
    return pid_list, data_x, get_go_list(pid_go_file, pid_list)


 # 通过Train_GO,获得所有的标签种类，后续二值化
def get_mlb(mlb_path: Path, labels=None, **kwargs) -> MultiLabelBinarizer:
    if mlb_path.exists():
        return joblib.load(mlb_path)
    mlb = MultiLabelBinarizer(sparse_output=True, **kwargs)
    mlb.fit(labels)
    joblib.dump(mlb, mlb_path)
    return mlb


def output_res(res_path: Path, pid_list, go_list, sc_mat):
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(res_path, 'w') as fp:
        for pid_, sc_ in zip(pid_list, sc_mat):
            for go_, s_ in zip(go_list, sc_):
                if s_ > 0.0:
                    print(pid_, go_, s_, sep='\t', file=fp)


def get_pid_go_mat(pid_go, pid_list, go_list):
    go_mapping = {go_: i for i, go_ in enumerate(go_list)}
    r_, c_, d_ = [], [], []
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go:
            for go_ in pid_go[pid_]:
                if go_ in go_mapping:
                    r_.append(i)
                    c_.append(go_mapping[go_])
                    d_.append(1)
    return ssp.csr_matrix((d_, (r_, c_)), shape=(len(pid_list), len(go_list)))


def get_pid_go_sc_mat(pid_go_sc, pid_list, go_list):
    sc_mat = np.zeros((len(pid_list), len(go_list)))
    for i, pid_ in enumerate(pid_list):
        if pid_ in pid_go_sc:
            for j, go_ in enumerate(go_list):
                sc_mat[i, j] = pid_go_sc[pid_].get(go_, -1e100)
    return sc_mat
# pid_蛋白质id ， data_y:蛋白质id对应标签矩阵， net——pid——map：所有蛋白质id，
# pid_list_：筛选出pid_list在 net_pid_map 中存在的蛋白质ID ，并将它们的索引、ID 和网络ID 分别存储在元组中
def get_ppi_idx(pid_list, data_y, net_pid_map):
    pid_list_ = tuple(zip(*[(i, pid, net_pid_map[pid])
                            for i, pid in enumerate(pid_list) if pid in net_pid_map]))
    assert pid_list_    # assert 可以有效地排除错误
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y

# 用于获取给定蛋白质列表中同源相互作用蛋白质的索引和 GO 类型
def get_homo_ppi_idx(pid_list, fasta_file, data_y, net_pid_map, net_blastdb, blast_output_path):
    blast_sim = blast(net_blastdb, pid_list, fasta_file, blast_output_path)
    pid_list_ = []
    for i, pid in enumerate(pid_list):  ## 遍历pid_list,根据 BLAST 结果和 net_pid_map 字典中的映射关系，找到其同源蛋白质的 ID
        blast_sim[pid][None] = float('-inf')
        pid_ = pid if pid in net_pid_map else max(blast_sim[pid].items(), key=lambda x: x[1])[0]
        if pid_ is not None:
            pid_list_.append((i, pid, net_pid_map[pid_]))
    pid_list_ = tuple(zip(*pid_list_))
    pid_list_ = (np.asarray(pid_list_[0]), pid_list_[1], np.asarray(pid_list_[2]))
    return pid_list_[0], pid_list_[1], pid_list_[2], data_y[pid_list_[0]] if data_y is not None else data_y
      # 如果不存在于net 找到blast其同源蛋白质
      # pid_list_[0]:存在于net中的蛋白质：pid_list索引，
      # pid_list_[1]:存在于net中的蛋白质：pid_listID，
      # pid_list_[2], 存在于net中的蛋白质 ：net索引
      # data_y ：标签矩阵