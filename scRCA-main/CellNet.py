#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:48:37 2021

@author: liuyan
"""
import torch
import torch.nn as nn


class cellNet(nn.Module):

    def __init__(self, input_dim,num_classes, init_weights=None):
        super(cellNet, self).__init__()
        # self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, num_classes),
        )
        if init_weights:
            self._initialize_weights()
    def forward(self, x):
        # x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)

        res = self.classifier[6](x)

        return res
    def forward_embedding(self, x):
        # x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        # x = self.classifier[4](x)
        # x = self.classifier[5](x)
        # x = self.classifier[6](x)
        return x
    def forward_pro(self, x):
        # x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)
        x = self.classifier[6](x)
        x=nn.functional.softmax(x,dim=1)
        return x