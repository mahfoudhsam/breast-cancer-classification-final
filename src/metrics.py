import numpy as np

def accuracy(y , y_hat):
    if len(y) == 0:
        return 0
    return np.sum(y == y_hat)/len(y)

def confusion_matrix(y,y_hat):
    tp = np.sum((y==1)&(y_hat==1))
    tn = np.sum((y==0)&(y_hat==0))
    fp = np.sum((y==0)&(y_hat==1))
    fn = np.sum((y==1)&(y_hat==0))
    return np.array([[tp,fp],
                     [fn,tn]])

def precision(y,y_hat):
    tp = np.sum((y==1)&(y_hat==1))
    fp = np.sum((y==0)&(y_hat==1))
    if (tp + fp) == 0:
        return 0
    return tp / (tp + fp)


def recall(y,y_hat):
    tp = np.sum((y==1)&(y_hat==1))
    fn = np.sum((y==1)&(y_hat==0))
    if (tp + fn) == 0:
        return 0
    return tp / (tp + fn)


def f1_score(y,y_hat):
    p = precision(y,y_hat)
    r = recall(y,y_hat)
    if (p + r) == 0:
        return 0
    return 2*p*r/(p+r)
