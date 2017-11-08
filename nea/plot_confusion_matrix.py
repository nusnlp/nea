# !/usr/bin/python
# -*- coding:utf-8 -*-  
# Author: Shengjia Yan
# Date: 2017-10-26
# Email: i@yanshengjia.com

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import logging
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def load_confusion_matrix(ref_score, pred_score):
    ref_file = open('../output/emb/rnn/prompt_1/fold_0/preds/dev_ref.txt', 'r')
    pred_file = open('../output/emb/rnn/prompt_1/fold_0/preds/dev_pred_49.txt', 'r')

    for ref in ref_file.readlines():
        ref = ref.strip('\n')
        ref = int(ref)
        ref_score.append(ref)
    for pred in pred_file.readlines():
        pred = pred.strip('\n')
        pred = float(pred)
        pred = round(pred)
        pred_score.append(pred)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        # 1. find out how many samples per class have received their correct label
        # 计算真正类别为k的样本被预测成各个类别的比例
        # e.g. 有25个样本的 true label 是 6，其中10个样本被预测为类别7，那么在混淆矩阵中 true label = 6 并且 predicted label = 7 的一个格子中的值为 0.4
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 2. get the precision (fraction of class-k predictions that have ground truth label k)
        # 计算预测的准确率
        # e.g. 预测为类别k的有12个，但其中只有9个的真正类别是k，那么准确率为 0.75
        # cm = cm.astype('float') / cm.sum(axis=0)[:, np.newaxis]
        
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
    ref_score = []		# true label
    pred_score = []		# predicted label

    load_confusion_matrix(ref_score, pred_score)

    nea_matrix = confusion_matrix(ref_score, pred_score)
    np.set_printoptions(precision=2)
    class_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(nea_matrix, classes=class_names, title='Confusion matrix, without normalization')
    plt.savefig('./unnormalized_cm.png')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(nea_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.savefig('./normalized_cm.png')
    
    # plt.show()

if __name__ == '__main__':
    main()



