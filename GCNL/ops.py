# operation lib

import numpy as np
from sklearn.metrics import roc_curve, auc
import warnings
import scipy.sparse as ssp
from sklearn.metrics import average_precision_score as aupr

def accuracy(preds, labels):
    return (100.0 * np.sum(np.argmax(preds, 1) == np.argmax(labels, 1))
          / preds.shape[0])

def RMSE(p, y):
    N = p.shape[0]
    diff = p - y
    return np.sqrt((diff**2).mean())

def ROC_AUC(p, y):

    fpr, tpr, th = roc_curve(y, p)
    _auc = auc(fpr, tpr)
    _roc = (fpr, tpr)
    return _roc, _auc


#对蛋白质序列热编码
def toOne_Hot(label, n_class):
    '''
    :param label: 原始标签
    :param n_class: 类别数量
    :param indexStart: 标签是从0开始还是从1开始,默认是从0开始
    :return: 转换后的One-Host标签
    '''
    # np.eye:返回的是一个二维2的数组(N,M)，对角线的地方为1，其余的地方为0.
    # param:N:表示的是输出的行数 dtyoe:返回的数据类型
    # 对角线为1的n_class＊n_class的矩阵,对应n_class个标签
    labelList = np.eye(n_class, dtype=int)
    # print(labelList)
    # 创建空矩阵，与标签矩阵的行的数量一样，列为8  param1:行数 param2:列数 param3:返回的数据类型
    labelXOneHot = np.empty([len(label), n_class], dtype=int)

    for index in range(len(label)):
        # 转换我，其实是修改labelXOneHot矩阵。
        # labelXOneHot[index] = labelList[label[index,0]- indexStart]
        labelXOneHot[index] = labelList[label[index, 0]]

    return labelXOneHot  # 返回


def fmax(targets: ssp.csr_matrix, scores: np.ndarray):
    fmax_ = 0.0, 0.0
    for cut in (c / 100 for c in range(101)):
        cut_sc = ssp.csr_matrix((scores >= cut).astype(np.int32))
        correct = cut_sc.multiply(targets).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p, r = correct / cut_sc.sum(axis=1), correct / targets.sum(axis=1)
            p, r = np.average(p[np.invert(np.isnan(p))]), np.average(r)
        if np.isnan(p):
            continue
        try:
            fmax_ = max(fmax_, (2 * p * r / (p + r) if p + r > 0.0 else 0.0, cut))
        except ZeroDivisionError:
            pass
    return fmax_

def pair_aupr(targets: ssp.csr_matrix, scores: np.ndarray, top=200):
    scores[np.arange(scores.shape[0])[:, None],
           scores.argpartition(scores.shape[1] - top)[:, :-top]] = -1e100
    scores[~np.isfinite(scores)] = 0
    return aupr(targets.toarray().flatten(), scores.flatten())
