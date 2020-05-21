# code by chenchiwei
# -*- coding: UTF-8 -*- 
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
import lightgbm
from lightgbm.sklearn import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import entropy
# from gensim.models import Word2VecF
import time
import gc
pd.set_option('display.max_columns', None)


# H 测试样本分类结果
# TrainS 原训练样本 np数组
# TrainA 辅助训练样本
# LabelS 原训练样本标签
# LabelA 辅助训练样本标签
# Test  测试样本
# N 迭代次数
def tradaboost(trans_S, trans_A, label_S, label_A, test, test_label, N):
    trans_data = np.concatenate((trans_A, trans_S), axis=0)
    trans_label = np.concatenate((label_A, label_S), axis=0)

    row_A = trans_A.shape[0]
    row_S = trans_S.shape[0]
    row_T = test.shape[0]

    test_data = np.concatenate((trans_data, test), axis=0)
    test_label = np.concatenate((trans_label, test_label), axis=0)

    # 初始化权重
    weights_A = np.ones([row_A, 1]) / row_A
    weights_S = np.ones([row_S, 1]) / row_S
    weights = np.concatenate((weights_A, weights_S), axis=0)

    bata = 1 / (1 + np.sqrt(2 * np.log(row_A) / N))

    # 存储每次迭代的标签和bata值？
    bata_T = np.zeros([1, N])
    result_label = np.ones([row_A + row_S + row_T, N])

    predict = np.zeros([row_T])
    predict_prob = np.zeros([row_T, 2])

    print('params initial finished.')
    trans_data = np.asarray(trans_data, order='C')
    trans_label = np.asarray(trans_label, order='C')
    test_data = np.asarray(test_data, order='C')

    for i in range(N):
        P = calculate_P(weights, row_A, row_S)

        result_label[:, i] = train_classify(trans_data, trans_label, row_A, row_S, 
                                test_data, test_label, P)
        print('result,', result_label[:, i], row_A, row_S, i, result_label.shape)

        error_rate = calculate_error_rate(label_S, result_label[row_A:row_A + row_S, i],
                                          weights[row_A:row_A + row_S, :])
        print('Error rate:', error_rate)
        if error_rate > 0.5:
            error_rate = 0.5
        if error_rate == 0:
            N = i
            break  # 防止过拟合
            # error_rate = 0.001

        bata_T[0, i] = error_rate / (1 - error_rate)

        # 调整源域样本权重
        for j in range(row_S):
            weights[row_A + j] = weights[row_A + j] * np.power(bata_T[0, i],
                                              (-np.abs(result_label[row_A + j, i] - label_S[j])))

        # 调整辅域样本权重
        for j in range(row_A):
            weights[j] = weights[j] * np.power(bata, np.abs(result_label[j, i] - label_A[j]))
    # print bata_T
    for i in range(row_T):
        # 跳过训练数据的标签
        left = np.sum(
            result_label[row_A + row_S + i, int(np.ceil(N / 2)):N] * np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))
        right = 0.5 * np.sum(np.log(1 / bata_T[0, int(np.ceil(N / 2)):N]))

        predict_prob[i] = [right/(left+right), left/(left+right)]
        if left >= right:
            predict[i] = 1
        else:
            predict[i] = 0
            # print left, right, predict[i]

    return predict_prob, predict


def calculate_P(weights, row_A, row_S):
    total = np.sum(weights)
    return np.asarray(weights / total, order='C')


def train_classify(trans_data, trans_label, row_A, row_S, test_data, test_label, P):
# #     lgb_params = {'objective': 'binary', 'metric': 'auc', 'verbosity': -1, 'boosting_type': 'gbdt', 'n_estimators': 2000,
# #               'lambda_l1': 9.870404094446918, 'lambda_l2': 0.0012884446637229194, 'num_leaves': 3, 
# #               'feature_fraction': 0.7, 'bagging_fraction': 0.7205112042938184, 'bagging_freq': 3, 'min_child_samples': 20}
#     trans_size = row_A + row_S
#     clf = LogisticRegression()
# #     clf = LGBMClassifier(**lgb_params)
# #     clf.fit(trans_data, trans_label, sample_weight=P[:, 0]*row_A, early_stopping_rounds=100, 
# #           eval_metric="auc", eval_set=[(test_data[trans_size:, :], test_label[trans_size:])])
#     clf.fit(trans_data, trans_label)
#     print(P[row_A:, 0])
#     print(trans_data[row_A:, :])
#     print(trans_label[row_A:])
#     print(test_data[trans_size:, :])
#     print(test_label[trans_size:])

    clf = tree.DecisionTreeClassifier(max_features="log2", max_depth=6, min_samples_leaf=8)
    clf.fit(trans_data, trans_label, sample_weight=P[:, 0])
    #return clf.predict(test_data)
    return clf.predict_proba(test_data)[:, 1]
    # return clf.predict(test_data)


def calculate_error_rate(label_R, label_H, weight):
    total = np.sum(weight)

    print(weight[:, 0] / total)
    print(np.abs(label_R - label_H))
    return np.sum(weight[:, 0] * np.abs(label_R - label_H) / total)
