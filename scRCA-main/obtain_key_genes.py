#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 10:43:10 2021

@author: liuyan
"""
from CellNet import cellNet
import torch
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
from model import CNN
import argparse, sys
import numpy as np
import datetime
import shutil
from numpy import *
from processing_celldata import read_cell,CellDataset
from loss import loss_coteaching
scRCA_g = cellNet(17159,7)
scRCA_g=torch.load("weights/ID_symmetric_s0.15.pt")
platform="ID"
DataPath="paper_data/Inter-dataset/PbmcBench/inDrop/iD_pbmc1.csv"
LabelsPath="paper_data/Inter-dataset/PbmcBench/inDrop/iD_pbmc1Labels.csv"
CV_RDataPath="paper_data/Inter-dataset/PbmcBench/inDrop/iD_pbmc1_folds.RData"
#OutputDir="test"
train_x,train_label,y_train,test_x,test_label,y_test_cell_name,test_name,gene_names,cell_typeindex=read_cell(DataPath, LabelsPath, CV_RDataPath, platform)
scRCA_g.eval()
pro=scRCA_g.forward_pro(torch.tensor(test_x,dtype=torch.float32).cuda()).cpu().detach().numpy()
# pred_label=np.argmax(pro,axis=1)
# from sklearn.metrics import accuracy_score
# accuracy_score(pred_label, test_label, normalize=False)
# from sklearn.metrics import confusion_matrix
# confusion_matrix(pred_label, test_label)
import lime
import lime.lime_tabular
# targets =  ['0', '1', '2', '3', '4', '5','6','7','8']
# zeisel=pd.read_csv(file_path)
# gene_names=pd.read_csv(file_path)["sample_labels"][1:]
def get_scRCA_pro(x):
    data=torch.tensor(x,dtype=torch.float32).cuda()
    pro=scRCA_g.forward_pro(data).cpu().detach().numpy()
    return pro
x=torch.tensor(test_x,dtype=torch.float32).cuda().cpu().detach().numpy()

pro=get_scRCA_pro(x)
explainer = lime.lime_tabular.LimeTabularExplainer(test_x,feature_names=gene_names,
                                                      class_names=y_test_cell_name,
                                                      discretize_continuous=True)
predicted_label=get_scRCA_pro(test_x).argmax(1)

temp_label=predicted_label[1]
exp = explainer.explain_instance((test_x[1]),get_scRCA_pro,
                                num_features=17159,
                                top_labels=7,
                                num_samples=5000)
temp = exp.as_list(label=temp_label)


