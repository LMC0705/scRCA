#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:13:22 2021

@author: liuyan
"""
import torch
import numpy as np
from data.utils import  noisify
import os
import sys
class CellDataset():
    def __init__(self, x,y,train=True,noise_type=None, noise_rate=0.2,random_state=0):
        self.train_data=x
        self.train_labels=y
        self.train=train
        self.noise_type=noise_type
        self.noise_rate=noise_rate
        
        
        if self.train:
            self.nb_classes=len(set(list(y.reshape(y.shape[0]))))
            
            if noise_type !='clean':
                # noisify train data
                # self.train_labels=np.asarray([[self.train_labels[i]] for i in range(len(self.train_labels))])
                self.train_noisy_labels, self.actual_noise_rate = noisify(train_labels=self.train_labels, noise_type=noise_type, noise_rate=noise_rate, random_state=random_state, nb_classes=self.nb_classes)
                self.train_noisy_labels=[i[0] for i in self.train_noisy_labels]
                _train_labels=[i[0] for i in self.train_labels]
                
                self.noise_or_not = np.transpose(self.train_noisy_labels)==np.transpose(_train_labels)
                # self.train_noisy_labels=self.train_noisy_labels.reshape(self.train_noisy_labels.shape[0])
        else:

            self.test_data=x
            self.test_labels=y
            
    def __getitem__(self, index):
        if self.train:
            if self.noise_type !='clean':
                inputs, targets = self.train_data[index], self.train_noisy_labels[index]
            else:
                inputs, targets = self.train_data[index], self.train_labels[index]
        else:
            inputs, targets = self.test_data[index], self.test_labels[index]
        return (torch.tensor(inputs,dtype=torch.float32)),torch.tensor(targets),index

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
import rpy2.robjects as robjects
import os 
import numpy as np
import pandas as pd
import time as tm

from sklearn.calibration import CalibratedClassifierCV
# import tripletcell
import argparse
def process_label_to_list(train_y):
    label=[]
    for i in range(len(train_y)):
        label_i=train_y[i][0]
        label.append(label_i)
    return label
from sklearn.preprocessing import LabelEncoder

def read_cell(DataPath, LabelsPath, CV_RDataPath, platform):
    robjects.r['load'](CV_RDataPath)
    nfolds = np.array(robjects.r['n_folds'], dtype = 'int')
    tokeep = np.array(robjects.r['Cells_to_Keep'], dtype = 'bool')
    col = np.array(robjects.r['col_Index'], dtype = 'int')
    col = col - 1 
    test_ind = np.array(robjects.r['Test_Idx'])
# print (test_ind)
    train_ind = np.array(robjects.r['Train_Idx'])
# read the data
    data = pd.read_csv(DataPath,index_col=0,sep=',')
    labels = pd.read_csv(LabelsPath, header=0,index_col=None, sep=',', usecols = col)

    label=process_label_to_list(labels.values)
    le=LabelEncoder()
    le.fit(label)
    numlabel=le.transform(label).reshape(len(label),1)
    labels = labels.iloc[tokeep]
    data = data.iloc[tokeep]
    
    cell_typeindex=list(le.classes_)


# normalize data
    data = np.log1p(data)
    tr_time=[]
    ts_time=[]
    truelab = []
    pred = []      
    for i in range(np.squeeze(nfolds)):
        print ("the",i,"-th nfolds")
        test_ind_i = np.array(test_ind[i], dtype = 'int') - 1
        train_ind_i = np.array(train_ind[i], dtype = 'int') - 1
        train=data.iloc[train_ind_i]
        test=data.iloc[test_ind_i]
        y_train=labels.iloc[train_ind_i]
        y_test=labels.iloc[test_ind_i] 
        y_test_cell_name=y_test          
        start=tm.time()
        test_name=list(test.index)
        #clf.fit(train, y_train)
        y_train=process_label_to_list(y_train.values)
        train_x=train.values
        label_number=len(list(set(y_train)))
        y_test=process_label_to_list(y_test.values)
        test_label=le.transform(y_test)
        train_label=le.transform(y_train)
        train_label=train_label.reshape(train_label.shape[0],1)
        # test_label=test_label.reshape(test_label.shape[0],1)
#         nb_classes=label_number
#         import data.utils
#         noisetrain_label,ss=data.utils.noisify_pairflip(train_label,0.2, random_state=0, nb_classes=nb_classes)
# #        noisetrain_label,ss=data.utils.noisify_multiclass_symmetric(train_label,0.6, random_state=0, nb_classes=nb_classes)
        
#         noisetrain_label=noisetrain_label.reshape(train_label.shape[0])
        test_x=test.values
        # print (y_test)
        # num_classes=len(set(noisetrain_label))
        # train_GcelNet.train_test_gcelcell(train_x, noisetrain_label,test_x,test_label,num_classes)
        return train_x,train_label,y_train,test_x,test_label,y_test_cell_name,test_name,list(test.columns),cell_typeindex
        # tripletcell.train_celltriplet(model,train, y_train,test,y_test,50,platform)
# platform="10Xv2"
# DataPath="paper_data/Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1.csv"
# LabelsPath="paper_data/Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1Labels.csv"
# CV_RDataPath="paper_data/Inter-dataset/PbmcBench/10Xv2/10Xv2_pbmc1_folds.RData"
# #OutputDir="test"
# train_x,train_label,test_x,test_label=read_cell(DataPath, LabelsPath, CV_RDataPath, platform)
# train_dataset=CellDataset(train_x, train_label,train=True,noise_type="pairflip",noise_rate=0.2)
# test_dataset=CellDataset(test_x, test_label,train=False,noise_type="pairflip",noise_rate=0.2)
