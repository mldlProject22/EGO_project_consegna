from torch.nn.modules.loss import TripletMarginLoss
from TA3Nmodel import ta3n_model
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
from loaderFeat import loaderFeat
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
import sklearn as sk
from sklearn import metrics
import random
from datetime import datetime
import optuna


args = parser.parse_args()

aggregation=args.aggregation
domain_shift = args.shift
domain_source = domain_shift.split('-')[0]
domain_train=domain_source+"-"+domain_source
domain_target= domain_shift.split('-')[1]
n_classes = args.n_classes
place_adv = args.place_adv   # relation / video / frame ----> relation a Yes solo con trn-m
batch=args.batch_size
modality=args.modality
trials=args.trials
use_attn=args.use_attn

np.random.seed(23052022)
torch.manual_seed(23052022)
random.seed(8062022)


def main():

  study = optuna.create_study( study_name='GRID SEARCH',direction="maximize", sampler=optuna.samplers.TPESampler(seed = 8062022), pruner=optuna.pruners.MedianPruner())
  study.optimize(objective, n_trials=trials)
  
  best_trial = study.best_trial

  print('***BEST PARAMETERS***')
  print('Accuracy: ', study.best_value)
  for key, value in best_trial.params.items():
    print("{}: {}".format(key, value))

  f = open("/content/EGO_Project/step_4/results.txt", "a")
  f.write("\n")
  f.write(str(datetime.now()))
  f.write("\n")
  f.write("shift "+domain_shift)
  f.write("\n")
  f.write("aggregation type "+aggregation)
  f.write("\n")
  f.write("adversarial layers [relation, video, frame] "+str(place_adv))
  f.write("\n")
  f.write("batch size "+str(batch))
  f.write("\n")
  f.write("epochs "+str(args.epochs))
  f.write("\n")
  f.write("attention mechanism "+str(args.use_attn))
  f.write("\n")
  f.write("learning rate "+str(args.lr))
  f.write("\n")
  f.write("beta "+str(args.beta))
  f.write("\n")
  f.write("gamma "+str(args.gamma))
  f.write("\n")
  f.write("best params "+str(best_trial.params.items()))
  f.write("\n")
  f.write("best accuracy "+str(study.best_value))
  f.write("\n")
  f.close()


def objective(trial):

  params = { 'beta' : trial.suggest_loguniform('beta', 0.1,0.9),
             'loss' : trial.suggest_loguniform('loss', 0.1,0.9)}

  top1_val = test(params, trial)

  return top1_val





def test(params,trial):

  model = ta3n_model(frame_aggregation=aggregation,modality=modality).cuda()   #avgpool or trn-m

  optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr)
  criterion = torch.nn.CrossEntropyLoss().cuda()
  criterion_domain = torch.nn.CrossEntropyLoss().cuda()

  beta = [params['beta'],params['beta'],params['beta']]
  loss_weights=[params['loss'],params['loss'],params['loss']]
  gamma = args.gamma
  mu = 0
  epochs=args.epochs


  #print('*** TRAINING STATS ***')
  for ep in range(epochs):

    train_loss = train(model, criterion, criterion_domain, optimizer, ep, beta, gamma, mu,loss_weights)
    #print(ep,' Loss: ', train_loss)
    #log_gradients_in_model(model,logger,ep)
    
    if ep== epochs-1: 
      top1_val, top5_val, val_loss = validate( model, criterion, n_classes)
      #print('*** VALIDATION STATS ***')
      #print('Top1: ', top1_val, 'Top5: ', top5_val, 'Loss: ', val_loss)

  trial.report(top1_val, ep)
  if trial.should_prune():
    raise optuna.exceptions.TrialPruned()

  return top1_val
  

def train(model, criterion, criterion_domain, optimizer, ep, beta, gamma, mu,loss_weights):

  losses_a=AverageMeter()
  losses_c=AverageMeter()
  losses=AverageMeter()
  top1=AverageMeter()
  losses_e_verb = AverageMeter()

  if aggregation=='avgpool': loss_weights[0]=0

  dataloader_source, dataloader_target = load_data(is_train = True)
  dataloader = enumerate(zip(dataloader_source, dataloader_target))
  
  #model.module.partialBN(False)

  model.train()
 
  for i, ((source_data, source_label), (target_data, target_label)) in dataloader :
   
    
    optimizer.zero_grad() 

    source_data = source_data.float().cuda(non_blocking=True)
    target_data = target_data.float().cuda(non_blocking=True)

    #print('dim source data', np.shape(source_data))
    #print('dim target data', np.shape(target_data))

    source_label = source_label.cuda(non_blocking=True)
    target_label =target_label.cuda(non_blocking=True)

    #print('dim source label', np.shape(source_label))
    #print('dim target label', np.shape(target_label))

    out_source, pred_domain_source, feat_source, out_target, pred_domain_target, feat_target, attn_source, attn_target = model(source_data,target_data, beta = beta,mu =0, reverse =  False)

    attn_source, out_source, out_source_2, pred_domain_source, feat_source = removeDummy(attn_source, out_source, out_source, pred_domain_source, feat_source, out_source.size(0))
    attn_target, out_target, out_target_2, pred_domain_target, feat_target = removeDummy(attn_target, out_target, out_target, pred_domain_target, feat_target, out_source.size(0))

    #print ("DIMMMMMM",out_source.size())
    #print (out_source)
    #out_source=out_source.cuda()

    loss=criterion(out_source,source_label) #classification

    losses_c.update(loss.item(), out_source.size(0))


    #adversarial loss 

    loss_adversarial = 0
    pred_domain_all = []
    pred_domain_target_all = []

    for l in range(len(place_adv)): #place_adv è un vettore binario che contiene indicazioni sulle adversarial presenti
      if place_adv[l] == 'Y':

        # reshape the features (e.g. 128x5x2 --> 640x2)
        pred_domain_source_single = pred_domain_source[l].view(-1, pred_domain_source[l].size()[-1])
        pred_domain_target_single = pred_domain_target[l].view(-1, pred_domain_target[l].size()[-1])

        # prepare domain labels
        source_domain_label = torch.zeros(pred_domain_source_single.size(0)).long()
        target_domain_label = torch.ones(pred_domain_target_single.size(0)).long()
        domain_label = torch.cat((source_domain_label,target_domain_label),0)

        domain_label = domain_label.cuda(non_blocking=True)

        pred_domain = torch.cat((pred_domain_source_single, pred_domain_target_single),0)
        pred_domain_all.append(pred_domain)
        pred_domain_target_all.append(pred_domain_target_single)

        
        loss_adversarial_single = criterion_domain(pred_domain, domain_label)
        loss_adversarial += loss_weights[l]*loss_adversarial_single


    if(place_adv!=['N','N','N']):
      losses_a.update(loss_adversarial.item(), pred_domain.size(0))
      loss += loss_adversarial

    # 3. attentive entropy loss
    if use_attn != 'none':

      loss_entropy_verb = attentive_entropy(torch.cat((out_source, out_target),0), pred_domain_all[1])
     
      losses_e_verb.update(loss_entropy_verb.item(), out_target.size(0))
    
      loss += gamma * loss_entropy_verb

     
    acc1,acc5=accuracy(out_source.data,source_label,topk=(1,5))

    top1.update(acc1.item(),out_source.size(0))

    losses.update(loss.item())
    

    
    loss.backward()
    optimizer.step()


  return losses_c.avg #loss di classificazione asscoiata all'ultima epoca


def validate(model, criterion, num_class):  #val_loader sarà dataloader_target

  losses = AverageMeter()
  top1_verb = AverageMeter()
  top5_verb = AverageMeter()

  val_loader = load_data(is_train = False)

  
  # switch to evaluate mode
  model.eval()


  for i, (val_data, val_label) in enumerate(val_loader):

    

    val_size_ori = val_data.size()  # original shape
    batch_val_ori = val_size_ori[0]
   

    # add dummy tensors to keep the same batch size for each epoch (for the last epoch)
    if batch_val_ori < args.batch_size:
      val_data_dummy = torch.zeros(args.batch_size - batch_val_ori, val_size_ori[1], val_size_ori[2])
      val_data = torch.cat((val_data, val_data_dummy))


    val_label_verb = val_label.cuda(non_blocking=True)
    val_data = val_data.float().cuda()

    with torch.no_grad():

      # compute output
      _, _, _, out_val, pred_domain_val, feat_val, _, attn_val = model(val_data,val_data, beta = args.beta, mu =0, reverse =  False)

      # ignore dummy tensors
      attn_val , out_val, out_val_2, pred_domain_val, feat_val = removeDummy( attn_val , out_val, out_val, pred_domain_val, feat_val, batch_val_ori)


      # measure accuracy and record loss
    
      loss_verb = criterion(out_val, val_label_verb)
    
      loss = loss_verb  # 0.5*(loss_verb+loss_noun)
    
      prec1_verb, prec5_verb = accuracy(out_val.data, val_label_verb, topk=(1, 5))

      #acc_nostra = multi_acc(out_val.data, val_label_verb)
      
      losses.update(loss.item(), out_val.size(0))

      top1_verb.update(prec1_verb.item(), out_val.size(0))
      top5_verb.update(prec5_verb.item(), out_val.size(0))

      #top1_verb.update(acc_nostra.item(), out_val.size(0))

  return top1_verb.avg, top5_verb.avg, losses.avg

    

  
## DEF DATA LOADER
def load_data(is_train):


  data_path_source_train = '/content/drive/MyDrive/Progetto_2022/Features/' + domain_train + '_train.pkl'
  data_path_target_train = '/content/drive/MyDrive/Progetto_2022/Features/' + domain_shift + '_train.pkl'

  data_path_test = '/content/drive/MyDrive/Progetto_2022/Features/' + domain_shift + '_test.pkl'

  dataset_path_source_train = '/content/EGO_Project/train_val/'+ domain_source +'_train.pkl'
  dataset_path_target_train = '/content/EGO_Project/train_val/'+ domain_target +'_train.pkl'

  dataset_path_test = '/content/EGO_Project/train_val/'+ domain_target +'_test.pkl'
  
  source_ds = loaderFeat(data_path_source_train,dataset_path_source_train,modality)
  target_ds = loaderFeat(data_path_target_train,dataset_path_target_train,modality)

  test_ds = loaderFeat(data_path_test,dataset_path_test,modality )

  len_source = source_ds.__len__()
  len_target = target_ds.__len__()
  num_samples = min(len_source,len_target)

  source_ds = source_ds.sampling(num_samples)
  target_ds = target_ds.sampling(num_samples)
  source_sampler = torch.utils.data.sampler.RandomSampler(source_ds)
  target_sampler = torch.utils.data.sampler.RandomSampler(target_ds)
  
  dataloader_source = DataLoader(source_ds,batch_size = args.batch_size, sampler = source_sampler,shuffle = False, num_workers = 1)
  dataloader_target = DataLoader(target_ds, batch_size = args.batch_size ,sampler = target_sampler ,shuffle = False , num_workers = 1)

  test_sampler = torch.utils.data.sampler.RandomSampler(test_ds)

  dataloader_test = DataLoader(test_ds, batch_size = 128 ,sampler = test_sampler ,shuffle = False , num_workers = 1)

  if(is_train): return dataloader_source, dataloader_target
  else: return dataloader_test

################## attentive entropy loss (source + target) #####################
def attentive_entropy(pred, pred_domain):
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)

    # attention weight
    entropy = torch.sum(-softmax(pred_domain) * logsoftmax(pred_domain), 1)
    weights = 1 + entropy

    # attentive entropy
    loss = torch.mean(weights * torch.sum(-softmax(pred) * logsoftmax(pred), 1))
    return loss

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def multi_acc(y_pred, y_test):
    #print(y_pred.size(),y_test.size())
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc

def removeDummy(attn, out_1, out_2, pred_domain, feat, batch_size):
  attn = attn[:batch_size]
  if isinstance(out_1, (list, tuple)):
    out_1 = (out_1[0][:batch_size], out_1[1][:batch_size])
  else:
    out_1 = out_1[:batch_size]
  out_2 = out_2[:batch_size]
  pred_domain = [pred[:batch_size] for pred in pred_domain]
  feat = [f[:batch_size] for f in feat]

  return  attn, out_1, out_2, pred_domain, feat

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