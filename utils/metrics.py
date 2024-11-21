from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score, average_precision_score
import numpy as np
from scipy.special import expit
import torch

def cal_rocauc(y_true, y_pred):
    rocauc_list = []
    y_pred = expit(y_pred)
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive and negetive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan
            is_valid = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))
        else:
            rocauc_list.append(np.nan)

    return np.array(rocauc_list)


def cal_aup(y_true, y_scores):
    aup = []
    y_prob = expit(y_scores)
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_valid = y_true[:, i] == y_true[:, i]
            aup.append(average_precision_score(y_true=y_true[is_valid, i], y_score=y_prob[is_valid, i]))
        else:
            aup.append(np.nan)
            
    return np.array(aup)

def cal_auc(y_true, y_pred):
    auc_list = []
    
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive and negetive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_valid = y_true[:, i] == y_true[:, i]
            fpr, tpr, _ =  roc_curve(y_true=y_true[is_valid, i], y_score=y_pred[is_valid, i])
            auc_list = auc(fpr, tpr)
        else:
            auc_list.append(np.nan)
    
    return np.array(auc_list)


def cal_acc(y_true, y_pred):
    acc_list = []
    
    y_pred[y_pred < 0.5] = 0
    y_pred[y_true >= 0.5] = 1
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive and negetive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_valid = y_true[:, i] == y_true[:, i]
            acc_list = accuracy_score(y_true[is_valid, i], y_pred[is_valid, i])
        else:
            acc_list.append(np.nan)
    return np.array(acc_list)

    
def topAcc(top_n, x1, x2, mask):
    # if x1 is Z_g and x2 is Z_d, it's: given a drug structure, find the most possible gene expression
    x1 = x1[mask, :]
    x2 = x2[mask, :]
    batch_size, _ = x1.size()
    x1 = x1 / x1.norm(dim = -1, keepdim = True)
    x2 = x2 / x2.norm(dim = -1, keepdim = True)
    smilarity = torch.matmul(x1, x2.T)
    argsort_sim = smilarity.argsort(dim=1, descending=True).argsort(dim=1) # argsort by row
    argsort_pos = argsort_sim[range(batch_size), range(batch_size)]
    print(argsort_pos[:10])
    return (argsort_pos < top_n).float().mean()


def topnAcc(x1, x2, mask):
    # if x1 is Z_g and x2 is Z_d, it's: given a drug structure, find the most possible gene expression
    x1 = x1[mask]
    x2 = x2[mask]
    batch_size = x1.shape[0]
    sim_matrix = torch.einsum('ik,jk->ij', x1, x2)
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    sim_matrix = sim_matrix / torch.einsum('i,j->ij', x1_abs, x2_abs)
    argsort_sim = sim_matrix.argsort(dim=1, descending=True).argsort(dim=1) # argsort by row
    argsort_pos = argsort_sim[range(batch_size), range(batch_size)]
    return (argsort_pos < 1).float().mean(), (argsort_pos < 5).float().mean(), (argsort_pos < 10).float().mean()
