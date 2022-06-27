import torch.utils.data as data
from torch.utils.data import DataLoader
#from utils.utils import AverageMeter
import sklearn as sk
import os
import os.path
import numpy as np
from numpy.random import randint
import torch
import pickle
import pandas as pd
import torch.nn as nn
from loaderFeat import loaderFeat
from args import parser
import I3Dclassifier 
import TRNclassifier


args = parser.parse_args()

modality = args.modality
domain_shift = args.shift
domain = domain_shift.split('-')[0]
n_classes = args.n_classes
model_type = args.model

data_path_feat_train = '/content/drive/MyDrive/Progetto_2022/Pre-extracted_feat/' + modality + '/ek_' + model_type.lower() + '/' + domain_shift + '_train.pkl'
data_path_feat_test = '/content/drive/MyDrive/Progetto_2022/Pre-extracted_feat/' + modality + '/ek_' + model_type.lower() + '/' + domain_shift + '_test.pkl'

dataset_path_train = '/content/EGO_Project/train_val/'+ domain +'_train.pkl'
dataset_path_test = '/content/EGO_Project/train_val/'+ domain +'_test.pkl'

def main():

  np.random.seed(23052022)
  torch.manual_seed(23052022)

  train_ds = loaderFeat(data_path_feat_train, modality ,dataset_path_train)
  test_ds = loaderFeat(data_path_feat_test,modality,dataset_path_test)

  train_dataloader = DataLoader(train_ds, batch_size = 16, shuffle = True , num_workers = 1)
  test_dataloader = DataLoader(test_ds, batch_size = 16, shuffle = False, num_workers = 1)

  if(model_type == 'i3d'): model = I3Dclassifier.i3d_classifier(input_size = 5*1017, n_classes = n_classes).cuda()
  else: model = TRNclassifier.RelationModuleMultiScaleWithClassifier(img_feature_dim = 2048, num_frames = 5, num_class = n_classes).cuda()

  class_weights = sk.utils.class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(train_ds.target()), y = train_ds.target())
  class_weights = torch.tensor(class_weights, dtype = torch.float).cuda()

  loss_crit = nn.CrossEntropyLoss(weight = class_weights)
  optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
  epochs = 13 #aumentare
  print("Start training......")

  for ep in range(epochs):
    loss_train = AverageMeter()
    top1_acc_train = AverageMeter()

    model.train()
    # activate the training in all modules (when training, DropOut is active, BatchNorm updates itself)
    # (when not training, BatchNorm is freezed, DropOut disabled)


    for data,labels in train_dataloader:
      data,labels = data.float().cuda(),  labels.long().cuda()
      optimizer.zero_grad()
      target = model(data)
      loss = loss_crit(target,labels)
      loss.backward()
      optimizer.step()
      loss_train.update(loss.item(), data.size(0))  # GIUSTO ????? METTERE DATA SIZE COSì ABBIAMO LA MEDIA PER EPOCH 
      train_accuracy = multi_acc(target,labels)
      top1_acc_train.update(train_accuracy.item(), data.size(0)) #####

    valid_loss = AverageMeter()
    top1_acc_test = AverageMeter()

    print("Start testing......")
    model.eval()
    for data,labels in test_dataloader:
      data,labels = data.float().cuda(),  labels.long().cuda()
      target = model(data)
      loss = loss_crit(target,labels)  
      valid_acc = multi_acc(target,labels)
      valid_loss.update(loss.item(), data.size(0))  ####
      top1_acc_test.update(valid_acc.item(), data.size(0))  #####
      
      
    print('Epoch: ' , ep)
    print('Training loss: ', loss_train.avg) 
    print('Top 1 accuracy Test :', top1_acc_test.avg)
    print('Validation loss: ', valid_loss)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count



if __name__ == '__main__':
  main()
