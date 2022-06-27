import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import torch
import pickle
import pandas as pd
import torch.nn as nn
from args import parser

class i3d_classifier(nn.Module):

  
  def __init__(self,input_size, n_classes,p):
    super().__init__()
    args = parser.parse_args()
    
    self.avg_pool = nn.AvgPool1d(kernel_size = 8, stride = 1, padding = 0)
    self.bn = nn.BatchNorm1d(5, eps = 0.001, momentum= 0.01)
    self.flatten = nn.Flatten()
    self.dropout = nn.Dropout(p)
    self.linear = nn.Linear(in_features = input_size, out_features = n_classes)
                                          #input size = 5*1017 perchè deve essere un divisore di 5075 e che dia 5 
  def forward(self,x):
    x = self.avg_pool(x)
    x = self.bn(x)
    x = self.flatten(x)
    x = self.dropout(x)
    x = self.linear(x)
    print(x)
    return x
