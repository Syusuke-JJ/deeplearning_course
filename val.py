import numpy as np
from fullyconnect import SoftmaxLossLayer

def softmax(x):
    max_x = np.max(x, axis=1).reshape(-1, 1)
    x = np.exp(x - max_x) / np.sum(np.exp(x - max_x), axis=1, keepdims=True)
    return x

def crossEntropyLoss(y_hat, y):
    y_hat = softmax(y_hat)
    l = 0
    for i, j in enumerate(y):
        l += -np.log(y_hat[i, j] + 1e-05)
    return l / len(y)

def validate(iter_,net):
    accuracy=0
    loss_list=[]
    for X,y in iter_:
        pro_y=net(X)
        loss=crossEntropyLoss(pro_y,y)
        accuracy+=(np.argmax(pro_y,axis=1)==y).sum()
        loss_list.append(loss.item())

    return accuracy/len(iter_),np.mean(loss_list)