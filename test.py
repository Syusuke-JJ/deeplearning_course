import numpy as np
from mlxtend.data import loadlocal_mnist
from dataloader import DataLoader
from fullyconnect import MLP2
import os

test_images ,test_labels= loadlocal_mnist(
    images_path="/root/YJ/NN/mnist_data/t10k-images-idx3-ubyte",
    labels_path="/root/YJ/NN/mnist_data/t10k-labels-idx1-ubyte")

input_size=784 #28*28
hidden=128
output_size=10
batch_size=64

test_iter=DataLoader(test_images,test_labels,batch_size)
mlp2=MLP2(input_size,hidden,output_size,lr=0,l2=0)
param_dir="/root/YJ/NN/param/mlp2-256hidden-0.1lr-0.0001l2.npy"
mlp2.load_model(param_dir)

accuracy=0
for X,y in test_iter:
    pro_y=mlp2(X)
    accuracy+=(np.argmax(pro_y,axis=1)==y).sum()
print('test accuracy:',accuracy/len(test_iter))
