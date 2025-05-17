import torch
import numpy as np

from tqdm import tqdm
from sklearn import metrics
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, accuracy_score
from data import create_dataloader
from data.datasets import*


def validate(model, opt):
    
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for data in tqdm(data_loader):
            input, _, tf_label, cmp_label = data

            input = input.cuda()
            tf_label = tf_label.cuda().float()

            tf_output = model(input)
            y_pred.extend(tf_output.sigmoid().flatten().tolist())
            y_true.extend(tf_label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    try:
        fpr, tpr, thresholds = metrics.roc_curve(y_true,y_pred, pos_label=1)
    except:
        # for the case when we only have one sample
        return acc, ap, 0, None, r_acc, f_acc, y_true, y_pred

    if np.isnan(fpr[0]) or np.isnan(tpr[0]):
        # for the case when all the samples within a batch is fake/real
        auc, eer = 0, None
    else:
        auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return acc, ap, auc, eer, r_acc, f_acc, y_true, y_pred
