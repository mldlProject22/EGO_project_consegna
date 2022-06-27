#references from

#Temporal Attentive Alignment for Large-Scale Video Domain Adaptation
#Chen, Min-Hung and Kira, Zsolt and AlRegib, Ghassan and Yoo, Jaekwon and Chen, Ruxin and Zheng, Jian
#2019
#https://arxiv.org/abs/1907.12743


#Temporal Attentive Alignment for Video Domain Adaptation
#Chen, Min-Hung and Kira, Zsolt and AlRegib, Ghassan
#2019
#https://arxiv.org/abs/1905.10861

#https://github.com/jonmun/EPIC-KITCHENS-100_UDA_TA3N

import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import torch
import pickle
import pandas as pd
import torch.nn as nn
from torch.autograd import Function
from args import parser
import TRNmodule
from torch.nn.init import *
from torch.autograd import Function
import torch.nn as nn
import torchvision
import math
import torch.nn.functional as F

np.random.seed(23052022)
torch.manual_seed(23052022)

args = parser.parse_args()


# definition of Gradient Reversal Layer
class GradReverse(Function):
	@staticmethod
	def forward(ctx, x, beta):
		ctx.beta = beta
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		grad_input = grad_output.neg() * ctx.beta
		return grad_input, None

class ta3n_model(nn.Module):

  def __init__(self, n_classes=8,p=0.5,frame_aggregation='avgpool',modality="RGB"):
    super().__init__()

    self.modality=modality
    self.n_classes = n_classes
    self.train_segments=5
    self.p = p 
    self.frame_aggregation = frame_aggregation
    self.use_attn = args.use_attn
    self._prepare_DA(n_classes)
    

  def _prepare_DA(self, num_class): # convert the model to DA framework

    if self.modality=="All":
      	self.feature_dim = 2048*2
    else:
      self.feature_dim = 2048

    std = 0.001
    feat_shared_dim = self.feature_dim 
    feat_frame_dim = feat_shared_dim

    self.relu = nn.ReLU(inplace=True)
    self.dropout_i = nn.Dropout(p = self.p)
    self.dropout_v = nn.Dropout(p=self.p)

  
    # 1. shared feature layers
    self.fc_feature_shared_source = nn.Linear(self.feature_dim, feat_shared_dim)
    normal_(self.fc_feature_shared_source.weight, 0, std) 
    constant_(self.fc_feature_shared_source.bias, 0)
    
    # 2. frame-level feature layers
    self.fc_feature_source = nn.Linear(feat_shared_dim, feat_frame_dim)
    normal_(self.fc_feature_source.weight, 0, std)
    constant_(self.fc_feature_source.bias, 0)

    # 3. domain feature layers (frame-level)
    self.fc_feature_domain = nn.Linear(feat_shared_dim, feat_frame_dim)
    normal_(self.fc_feature_domain.weight, 0, std)
    constant_(self.fc_feature_domain.bias, 0)

    # 4. classifiers (frame-level)
    self.fc_classifier_source = nn.Linear(feat_frame_dim, num_class)  
    normal_(self.fc_classifier_source.weight, 0, std)
    constant_(self.fc_classifier_source.bias, 0)
             
    self.fc_classifier_domain = nn.Linear(feat_frame_dim, 2)
    normal_(self.fc_classifier_domain.weight, 0, std)
    constant_(self.fc_classifier_domain.bias, 0)

    self.bn_shared_S = nn.BatchNorm1d(feat_shared_dim)  # BN for the shared layers
    self.bn_shared_T = nn.BatchNorm1d(feat_shared_dim)
    self.bn_source_S = nn.BatchNorm1d(feat_frame_dim)  # BN for the source feature layers
    self.bn_source_T = nn.BatchNorm1d(feat_frame_dim)

    
  #------ aggregate frame-based features (frame feature --> video feature) ------#
  
    if self.frame_aggregation == 'trn-m':  
        self.num_bottleneck = 256
        self.TRN = TRNmodule.RelationModuleMultiScale(feat_shared_dim, self.num_bottleneck, self.train_segments)
        self.bn_trn_S = nn.BatchNorm1d(self.num_bottleneck)
        self.bn_trn_T = nn.BatchNorm1d(self.num_bottleneck)

  # ------ video-level layers (source layers + domain layers) ------#
    if self.frame_aggregation == 'avgpool': # 1. avgpool
        feat_aggregated_dim = feat_shared_dim
    else : # 4. trn
      feat_aggregated_dim = self.num_bottleneck
    
    feat_video_dim = feat_aggregated_dim

		# 1. source feature layers (video-level)
    self.fc_feature_video_source = nn.Linear(feat_aggregated_dim, feat_video_dim)
    normal_(self.fc_feature_video_source.weight, 0, std)
    constant_(self.fc_feature_video_source.bias, 0)

    self.fc_feature_video_source_2 = nn.Linear(feat_video_dim, feat_video_dim)
    normal_(self.fc_feature_video_source_2.weight, 0, std)
    constant_(self.fc_feature_video_source_2.bias, 0)

    # 2. domain feature layers (video-level)
    self.fc_feature_domain_video = nn.Linear(feat_aggregated_dim, feat_video_dim)
    normal_(self.fc_feature_domain_video.weight, 0, std)
    constant_(self.fc_feature_domain_video.bias, 0)

    # 3. classifiers (video-level)
    self.fc_classifier_video_source = nn.Linear(feat_video_dim, num_class)  
    normal_(self.fc_classifier_video_source.weight, 0, std)
    constant_(self.fc_classifier_video_source.bias, 0)

    self.fc_classifier_domain_video = nn.Linear(feat_video_dim, 2)
    normal_(self.fc_classifier_domain_video.weight, 0, std)
    constant_(self.fc_classifier_domain_video.bias, 0)

		# domain classifier for TRN-M
    if self.frame_aggregation == 'trn-m':
      self.relation_domain_classifier_all = nn.ModuleList()
      for i in range(self.train_segments-1):
        relation_domain_classifier = nn.Sequential(
          nn.Linear(feat_aggregated_dim, feat_video_dim),
          nn.ReLU(),
          nn.Linear(feat_video_dim, 2)
        )
        self.relation_domain_classifier_all += [relation_domain_classifier]

    self.bn_source_video_S = nn.BatchNorm1d(feat_video_dim)
    self.bn_source_video_T = nn.BatchNorm1d(feat_video_dim)
    self.bn_source_video_2_S = nn.BatchNorm1d(feat_video_dim)
    self.bn_source_video_2_T = nn.BatchNorm1d(feat_video_dim)

    self.alpha = torch.ones(1)

    # ------ attention mechanism ------#
    
    ## conventional attention
    if self.use_attn == 'general':
      self.attn_layer = nn.Sequential(
        nn.Linear(feat_aggregated_dim, feat_aggregated_dim),
        nn.Tanh(),
        nn.Linear(feat_aggregated_dim, 1)
        )

  def get_trans_attn(self, pred_domain):
      softmax = nn.Softmax(dim=1)
      logsoftmax = nn.LogSoftmax(dim=1)
      entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
      weights = 1 - entropy

      return weights

  def get_general_attn(self, feat):
    num_segments = feat.size()[1]
    feat = feat.view(-1, feat.size()[-1]) # reshape features: 128x4x256 --> (128x4)x256
    weights = self.attn_layer(feat) # e.g. (128x4)x1
    weights = weights.view(-1, num_segments, weights.size()[-1]) # reshape attention weights: (128x4)x1 --> 128x4x1
    weights = F.softmax(weights, dim=1)  # softmax over segments ==> 128x4x1

    return weights

  def get_attn_feat_frame(self, feat_fc, pred_domain): # not used for now
    if self.use_attn == 'TransAttn':
      weights_attn = self.get_trans_attn(pred_domain)
    elif self.use_attn == 'general':
      weights_attn = self.get_general_attn(feat_fc)

    weights_attn = weights_attn.view(-1, 1).repeat(1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 512)
    feat_fc_attn = (weights_attn+1) * feat_fc

    return feat_fc_attn

  def get_attn_feat_relation(self, feat_fc, pred_domain, num_segments):
    if self.use_attn == 'TransAttn':
      weights_attn = self.get_trans_attn(pred_domain)
    elif self.use_attn == 'general':
      weights_attn = self.get_general_attn(feat_fc)

    weights_attn = weights_attn.view(-1, num_segments-1, 1).repeat(1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 4 x 256)
    feat_fc_attn = (weights_attn+1) * feat_fc

    return feat_fc_attn, weights_attn[:,:,0]

  def forward(self, input_source, input_target, beta, mu, reverse,is_train):
    
    batch_source = input_source.size()[0]
    batch_target = input_target.size()[0]
    
    num_segments = 5
    feat_all_source = []
    feat_all_target = []
    pred_domain_all_source = []
    pred_domain_all_target = []

    feat_base_source = input_source.view(-1, input_source.size()[-1]) # e.g. 256 x 25 x 2048 --> 6400 x 2048
    feat_base_target = input_target.view(-1, input_target.size()[-1])  # e.g. 256 x 25 x 2048 --> 6400 x 2048
    
    feat_fc_source = self.fc_feature_shared_source(feat_base_source) #primo linear layer
    feat_fc_target = self.fc_feature_shared_source(feat_base_target)
  

    feat_fc_source, feat_fc_target = self.domainAlign(feat_fc_source, feat_fc_target, is_train, 'shared', self.alpha.item(), num_segments, 1)


    feat_fc_source = self.relu(feat_fc_source)
    feat_fc_target = self.relu(feat_fc_target)
    feat_fc_source = self.dropout_i(feat_fc_source)
    feat_fc_target = self.dropout_i(feat_fc_target)

    feat_all_source.append(feat_fc_source.view((batch_source, num_segments) + feat_fc_source.size()[-1:])) # reshape ==> 1st dim is the batch size
    feat_all_target.append(feat_fc_target.view((batch_target, num_segments) + feat_fc_target.size()[-1:]))
    
    pred_fc_domain_frame_source = self.domain_classifier_frame(feat_fc_source, beta) #predizioni domain    #primo adversarial frame level
    pred_fc_domain_frame_target = self.domain_classifier_frame(feat_fc_target, beta)

    pred_domain_all_source.append(pred_fc_domain_frame_source.view((batch_source, num_segments) + pred_fc_domain_frame_source.size()[-1:]))
    pred_domain_all_target.append(pred_fc_domain_frame_target.view((batch_target, num_segments) + pred_fc_domain_frame_target.size()[-1:]))

    pred_fc_source = self.fc_classifier_source(feat_fc_source) #predizioni classificazione
    pred_fc_target = self.fc_classifier_source(feat_fc_target)

    if self.frame_aggregation == 'avgpool':
	
      feat_fc_video_source = self.aggregate_frames(feat_fc_source, num_segments, pred_fc_domain_frame_source) #layer di avg pooling 
      feat_fc_video_target = self.aggregate_frames(feat_fc_target, num_segments, pred_fc_domain_frame_target)

      attn_relation_source = feat_fc_video_source[:,0] # assign random tensors to attention values to avoid runtime error
      attn_relation_target = feat_fc_video_target[:,0] # assign random tensors to attention values to avoid runtime error

    else:

      feat_fc_video_source = feat_fc_source.view((-1, num_segments) + feat_fc_source.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)  noi se batch 32: 160 x 20489 --> 32 x 5 x 2048
      feat_fc_video_target = feat_fc_target.view((-1, num_segments) + feat_fc_target.size()[-1:])  # reshape based on the segments (e.g. 640x512 --> 128x5x512)
      
      feat_fc_video_relation_source = self.TRN(feat_fc_video_source) # 128x5x512 --> 128x5x256 (256-dim. relation feature vectors x 5)
      feat_fc_video_relation_target = self.TRN(feat_fc_video_target) #noi 32 x 4 x 256 in entrambi

			# adversarial branch
      pred_fc_domain_video_relation_source = self.domain_classifier_relation(feat_fc_video_relation_source, beta)    #relation adversarial solo con trn-m
      pred_fc_domain_video_relation_target = self.domain_classifier_relation(feat_fc_video_relation_target, beta)

      if self.use_attn != 'none': # get the attention weighting
        feat_fc_video_relation_source, attn_relation_source = self.get_attn_feat_relation(feat_fc_video_relation_source, pred_fc_domain_video_relation_source, num_segments)
        feat_fc_video_relation_target, attn_relation_target = self.get_attn_feat_relation(feat_fc_video_relation_target, pred_fc_domain_video_relation_target, num_segments)
      else:
        attn_relation_source = feat_fc_video_relation_source[:,:,0] # assign random tensors to attention values to avoid runtime error
        attn_relation_target = feat_fc_video_relation_target[:,:,0] # assign random tensors to attention values to avoid runtime error

			# sum up relation features (ignore 1-relation)
      feat_fc_video_source = torch.sum(feat_fc_video_relation_source, 1)
      feat_fc_video_target = torch.sum(feat_fc_video_relation_target, 1)

	
    feat_all_source.append(feat_fc_video_source.view((batch_source,) + feat_fc_video_source.size()[-1:]))
    feat_all_target.append(feat_fc_video_target.view((batch_target,) + feat_fc_video_target.size()[-1:]))

		#=== source layers (video-level) ===#
    feat_fc_video_source = self.dropout_v(feat_fc_video_source)
    feat_fc_video_target = self.dropout_v(feat_fc_video_target)

    if reverse:
      feat_fc_video_source = GradReverse.apply(feat_fc_video_source, mu)
      feat_fc_video_target = GradReverse.apply(feat_fc_video_target, mu)

    pred_fc_video_source = self.fc_classifier_video_source(feat_fc_video_source)
    pred_fc_video_target = self.fc_classifier_video_source(feat_fc_video_target) 

    # only store the prediction from classifier 1 (for now)
    feat_all_source.append(pred_fc_video_source.view((batch_source,) + pred_fc_video_source.size()[-1:]))
    feat_all_target.append(pred_fc_video_target.view((batch_target,) + pred_fc_video_target.size()[-1:]))


    #=== adversarial branch (video-level) ===#
    pred_fc_domain_video_source = self.domain_classifier_video(feat_fc_video_source, beta)    #ultimo adversarial per tutti video level
    pred_fc_domain_video_target = self.domain_classifier_video(feat_fc_video_target, beta)

    pred_domain_all_source.append(pred_fc_domain_video_source.view((batch_source,) + pred_fc_domain_video_source.size()[-1:]))
    pred_domain_all_target.append(pred_fc_domain_video_target.view((batch_target,) + pred_fc_domain_video_target.size()[-1:]))

    # video relation-based discriminator
    if self.frame_aggregation == 'trn-m':
      num_relation = feat_fc_video_relation_source.size()[1]
      pred_domain_all_source.append(pred_fc_domain_video_relation_source.view((batch_source, num_relation) + pred_fc_domain_video_relation_source.size()[-1:]))
      pred_domain_all_target.append(pred_fc_domain_video_relation_target.view((batch_target, num_relation) + pred_fc_domain_video_relation_target.size()[-1:]))
    else:
      pred_domain_all_source.append(pred_fc_domain_video_source) # if not trn-m, add dummy tensors for relation features
      pred_domain_all_target.append(pred_fc_domain_video_target)

    #=== final output ===#
    output_source = self.final_output(pred_fc_source, pred_fc_video_source, num_segments) # select output from frame or video prediction
    output_target = self.final_output(pred_fc_target, pred_fc_video_target, num_segments)
    
    
    return output_source, pred_domain_all_source[::-1], feat_all_source[::-1], output_target, pred_domain_all_target[::-1], feat_all_target[::-1], attn_relation_source, attn_relation_target


  def aggregate_frames(self, feat_fc, num_segments, pred_domain):

    # 1. averaging
    feat_fc_video = feat_fc.view((-1, 1, num_segments) + feat_fc.size()[-1:])  # reshape based on the segments (e.g. 16 x 1 x 5 x 512)

    if self.use_attn == 'TransAttn': # get the attention weighting
      weights_attn = self.get_trans_attn(pred_domain)
      weights_attn = weights_attn.view(-1, 1, num_segments,1).repeat(1,1,1,feat_fc.size()[-1]) # reshape & repeat weights (e.g. 16 x 1 x 5 x 512)
      feat_fc_video = (weights_attn+1) * feat_fc_video

    feat_fc_video = nn.AvgPool2d([num_segments, 1])(feat_fc_video)  # e.g. 16 x 1 x 1 x 512
    feat_fc_video = feat_fc_video.squeeze(1).squeeze(1)  # e.g. 16 x 512

    return feat_fc_video

  def domain_classifier_frame(self, feat, beta):
    feat_fc_domain_frame = GradReverse.apply(feat, beta[2])
    feat_fc_domain_frame = self.fc_feature_domain(feat_fc_domain_frame)
    feat_fc_domain_frame = self.relu(feat_fc_domain_frame)
    pred_fc_domain_frame = self.fc_classifier_domain(feat_fc_domain_frame)

    return pred_fc_domain_frame

  def domain_classifier_video(self, feat_video, beta):
    feat_fc_domain_video = GradReverse.apply(feat_video, beta[1])
    feat_fc_domain_video = self.fc_feature_domain_video(feat_fc_domain_video)
    feat_fc_domain_video = self.relu(feat_fc_domain_video)
    pred_fc_domain_video = self.fc_classifier_domain_video(feat_fc_domain_video)

    return pred_fc_domain_video

  def domain_classifier_relation(self, feat_relation, beta):
      # 128x4x256 --> (128x4)x2
      #noi abbiamo batch x 4 x 256 vorremmo avere (batchx4)x2 quindi

    pred_fc_domain_relation_video = None
    for i in range(len(self.relation_domain_classifier_all)):
      feat_relation_single = feat_relation[:,i,:].squeeze(1) # 128x1x256 --> 128x256
      feat_fc_domain_relation_single = GradReverse.apply(feat_relation_single, beta[0]) # the same beta for all relations (for now)

      pred_fc_domain_relation_single = self.relation_domain_classifier_all[i](feat_fc_domain_relation_single)
    
      if pred_fc_domain_relation_video is None:
        pred_fc_domain_relation_video = pred_fc_domain_relation_single.view(-1,1,2)
      else:
        pred_fc_domain_relation_video = torch.cat((pred_fc_domain_relation_video, pred_fc_domain_relation_single.view(-1,1,2)), 1)
      
    pred_fc_domain_relation_video = pred_fc_domain_relation_video.view(-1,2)
    
    
    return pred_fc_domain_relation_video

  def final_output(self, pred, pred_video, num_segments):
    return pred_video

  def domainAlign(self, input_S, input_T, is_train, name_layer, alpha, num_segments, dim):
    input_S = input_S.view((-1, dim, num_segments) + input_S.size()[-1:])  # reshape based on the segments (e.g. 80 x 512 --> 16 x 1 x 5 x 512)
    input_T = input_T.view((-1, dim, num_segments) + input_T.size()[-1:])  # reshape based on the segments

    # clamp alpha
    alpha = max(alpha,0.5)

    # rearange source and target data
    num_S_1 = int(round(input_S.size(0) * alpha))
    num_S_2 = input_S.size(0) - num_S_1
    num_T_1 = int(round(input_T.size(0) * alpha))
    num_T_2 = input_T.size(0) - num_T_1

    if is_train and num_S_2 > 0 and num_T_2 > 0:
      input_source = torch.cat((input_S[:num_S_1], input_T[-num_T_2:]), 0)
      input_target = torch.cat((input_T[:num_T_1], input_S[-num_S_2:]), 0)
    else:
      input_source = input_S
      input_target = input_T

    # adaptive BN
    input_source = input_source.view((-1, ) + input_source.size()[-1:]) # reshape to feed BN (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
    input_target = input_target.view((-1, ) + input_target.size()[-1:])

    if name_layer == 'shared':
      input_source_bn = self.bn_shared_S(input_source)
      input_target_bn = self.bn_shared_T(input_target)
    elif 'trn' in name_layer:
      input_source_bn = self.bn_trn_S(input_source)
      input_target_bn = self.bn_trn_T(input_target)
    elif name_layer == 'temconv_1':
      input_source_bn = self.bn_1_S(input_source)
      input_target_bn = self.bn_1_T(input_target)
    elif name_layer == 'temconv_2':
      input_source_bn = self.bn_2_S(input_source)
      input_target_bn = self.bn_2_T(input_target)

    input_source_bn = input_source_bn.view((-1, dim, num_segments) + input_source_bn.size()[-1:])  # reshape back (e.g. 80 x 512 --> 16 x 1 x 5 x 512)
    input_target_bn = input_target_bn.view((-1, dim, num_segments) + input_target_bn.size()[-1:])  #

    # rearange back to the original order of source and target data (since target may be unlabeled)
    if is_train and num_S_2 > 0 and num_T_2 > 0:
      input_source_bn = torch.cat((input_source_bn[:num_S_1], input_target_bn[-num_S_2:]), 0)
      input_target_bn = torch.cat((input_target_bn[:num_T_1], input_source_bn[-num_T_2:]), 0)

    # reshape for frame-level features
    if name_layer == 'shared' or name_layer == 'trn_sum':
      input_source_bn = input_source_bn.view((-1,) + input_source_bn.size()[-1:])  # (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
      input_target_bn = input_target_bn.view((-1,) + input_target_bn.size()[-1:])
    elif name_layer == 'trn':
      input_source_bn = input_source_bn.view((-1, num_segments) + input_source_bn.size()[-1:])  # (e.g. 16 x 1 x 5 x 512 --> 80 x 512)
      input_target_bn = input_target_bn.view((-1, num_segments) + input_target_bn.size()[-1:])

    return input_source_bn, input_target_bn





