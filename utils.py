'''
Evaluation code is based on 'Searching Central Difference Convolutional Networks for Face Anti-Spoofing' By Zitong Yu & Zhuo Su, 2019
github: https://github.com/ZitongYu/CDCN/blob/master/CVPR2020_paper_codes/utils.py
'''

import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import sklearn
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from collections import defaultdict

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def accuracy(output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0/batch_size))
        return res

def get_err_threhold_cross_db(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0

    right_index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]

    return err, best_th, right_index

def performances_cross_db(prediction_scores, gt_labels, pos_label=1, verbose=True):

    data = [{'map_score': score, 'label': label} for score, label in zip(prediction_scores, gt_labels)]
    fpr, tpr, threshold = roc_curve(gt_labels, prediction_scores, pos_label=pos_label)

    val_err, val_threshold, right_index = get_err_threhold_cross_db(fpr, tpr, threshold)
    test_auc = auc(fpr, tpr)

    FRR = 1 - tpr    # FRR = 1 - TPR
    HTER = (fpr+FRR)/2.0    # error recognition rate &  reject recognition rate

    if verbose is True:
        print(f'AUC@ROC is {test_auc}, HTER is {HTER[right_index]}, APCER: {fpr[right_index]}, BPCER: {FRR[right_index]}, EER is {val_err}, TH is {val_threshold}')

    return test_auc, fpr[right_index], FRR[right_index], HTER[right_index]

def compute_video_score(video_ids, predictions, labels):
    predictions_dict, labels_dict = defaultdict(list), defaultdict(list)

    for i in range(len(video_ids)):
        video_key = video_ids[i]
        predictions_dict[video_key].append(predictions[i])
        labels_dict[video_key].append(labels[i])

    new_predictions, new_labels, new_video_ids = [], [], []

    for video_indx in list(set(video_ids)):
        new_video_ids.append(video_indx)
        scores = np.mean(predictions_dict[video_indx])

        label = labels_dict[video_indx][0]
        new_predictions.append(scores)
        new_labels.append(label)

    return new_predictions, new_labels, new_video_ids
