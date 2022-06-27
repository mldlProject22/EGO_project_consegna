import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint
import torch
import pickle
import pandas as pd

class loaderFeat(data.Dataset):


    def __init__(self, data_path_feat, modality, dataset_path, num_classes = 8):
        self.modality = modality
        try:
            with open(data_path_feat, "rb") as f:
                data = pickle.load(f)  
            self.data_feat = data['features'][modality]
            self.data_narrations = data['narration_ids']
            self.data = dict(zip(self.data_narrations, self.data_feat))
        except:
            raise Exception("Cannot read the feature data in the given pickle file {}".format(data_path_feat))

        try:
            with open(dataset_path, "rb") as f:
                dataset = pickle.load(f)  
            self.dataset = dataset
            self.labels = self.dataset['verb_class']
        except:
            raise Exception("Cannot read the dataset in the given pickle file {}".format(dataset_path))

        self.num_classes = num_classes

    def __len__(self):
      return len(self.dataset)

    def target(self):
      return np.array(self.labels)

    def __getitem__(self, i):
      row = self.dataset.iloc[i]
      label = row['verb_class']
      narration_ids = row['video_id'] + "_" + str(row['uid'])
      feature = self.data[narration_ids]
      return feature, label


if __name__ == '__main__':
  data = loaderFeat('/content/drive/MyDrive/Progetto_2022/Pre-extracted_feat/Flow/ek_i3d/D1-D1_train.pkl',\
   'Flow','/content/EGO_Project/train_val/D1_train.pkl')
  target = data.target()
  print(np.unique(target,return_counts = True))
  data.__getitem__(1)
  print(np.shape(data.data['P08_01_12915']))

