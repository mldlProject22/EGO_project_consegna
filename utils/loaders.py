import math
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import librosa
import os.path
import numpy as np
from numpy.random import randint
import pickle

class VideoDataset(data.Dataset):
    def __init__(self, list_file, modality, image_tmpl,
                 num_frames_per_clip, dense_sampling,
                 sample_offset=0, num_clips=1,
                 fixed_offset=False,
                 visual_path=None, flow_path=None, event_path=None,
                 mode='train', transform=None, args=None):

        self.load_cineca_data = os.environ["HOME"].split("/")[-1] == "abottin1"
        self.sync = args.sync
        self.num_frames = num_frames_per_clip
        self.num_clips = num_clips
        self.resampling_rate= args.resampling_rate
        self.sample_offset = sample_offset
        self.fixed_offset = fixed_offset
        self.dense_sampling = dense_sampling
        #self.audio_path = args.audio_path
        #self.audio_path = pickle.load(open(self.audio_path, 'rb'))

        self.modalities = modality  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.args = args
        self.factor = {"RGB": 1, "Spec":1, "Flow": 2, "Event": self.args.rgb4e}

        self.stride = {"RGB": 2, "Spec":1, "Flow": 1, "Event": 1}
        self.flow_path = flow_path
        self.visual_path = visual_path
        
        self.event_path = event_path
        self.list_file = list_file  # all files paths taken from .pkl file

        # data related
        self.num_clips = num_clips
        self.image_tmpl = image_tmpl  # filename format. (ex. 'img_{:010d}.jpg' for RGB)
        self.transform = transform  # pipeline of transforms

        self.video_list = [EpicVideoRecord(tup, self.args.rgb4e) for tup in self.list_file.iterrows()]


    def _sample_train(self, record, modality='RGB'):
        if self.dense_sampling[modality]:
            '''
            TO BE COMPLETED!
            '''

        else:
            '''
            TO BE COMPLETED!
            '''

        return indices

    def _get_train_indices(self, record):
        '''
        TO BE COMPLETED!
        '''
        return segment_indices

    def _get_val_indices(self, record, modality): 
      frames=[]
      num_frames_video = record.num_frames[modality]

      if self.dense_sampling[modality]:###DENSE
        if modality == 'RGB':
          if(num_frames_video<=31): ## CON SALTI
            if num_frames_video<=15: max_starting_point=1
            else: max_starting_point=num_frames_video-15
            random_starting_points = np.random.randint(0, max_starting_point, self.num_clips)
            for p in random_starting_points:
              clip = []
              for i in range(0,self.num_frames[modality]):
                clip.append(p+i)
              frames.append(clip)
          else: 
            max_starting_point = num_frames_video - 31
            random_starting_points = np.random.randint(0, max_starting_point, self.num_clips)
            for p in random_starting_points:
              clip = []
              for i in range(0,self.num_frames[modality]):
                clip.append(p+2*i)
              frames.append(clip)
        else: ## SENZA SALTI PER FLOW
          if(num_frames_video<=15): max_starting_point=num_frames_video
          else: max_starting_point = num_frames_video - 15
          random_starting_points = np.random.randint(0, max_starting_point, self.num_clips)
          for p in random_starting_points:
            clip = []
            for i in range(0,self.num_frames[modality]):
              clip.append(p+i)
            frames.append(clip)
        """
        if(self.num_clips == 1):
            frames = frames[0]
        """
      else: ###UNIFORM TSM
        tot_frames = self.num_clips * self.num_frames[modality]
        if num_frames_video > tot_frames:
          tick = (num_frames_video/float(tot_frames))
          offsets = [int(tick/2.0 + tick*x) for x in range(tot_frames)]
        else:
          offsets = [int(x) for x in range(num_frames_video)]
          for i in range(tot_frames-num_frames_video):
            offsets.append(num_frames_video - 1)
        frames = [x + 1 for x in offsets] 
        frames=np.reshape(frames,(5,5))
      return frames

    def __getitem__(self, index):

        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        record = self.video_list[index]
        

        if self.mode == 'train':
            segment_indices = self._get_train_indices(record)

        else:  # val or test case
            segment_indices = {}
            for m in self.modalities:
                segment_indices[m] = self._get_val_indices(record, m)


        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        return frames, label

    '''
                Audio-related
    '''

    def _log_specgram(self, audio, window_size=10, step_size=5, eps=1e-6):
        '''
        TO BE COMPLETED!
        '''

    def _extract_sound_feature(self, record, idx):
        '''
        TO BE COMPLETED!
        '''

        return self._log_specgram(samples)


    def get(self, modality, record, indices):
        images = list()
        if self.num_clips > 1:
          for frame_index in indices:
            for i in frame_index:
              p = int(i)
              frame = self._load_data(modality, record, p)
              images.extend(frame)
        
        else:
          if(np.shape(indices) == (1,16)): 
            indices = indices[0]
          for i in indices:
              p = int(i)
              frame = self._load_data(modality, record, p)
              images.extend(frame)

        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):

        if modality == 'RGB' or modality == 'RGBDiff':
            idx_untrimmed = record.start_frame + idx
            img = Image.open(os.path.join(self.visual_path, record.untrimmed_video_name,
                                          self.image_tmpl[modality].format(idx_untrimmed))).convert('RGB')
            return [img]
        elif modality == 'Flow':
            idx_untrimmed = (record.start_frame // 2) + idx
           
            try:
                x_img = Image.open(os.path.join(self.flow_path, record.untrimmed_video_name,
                                                self.image_tmpl[modality].format('x', idx_untrimmed))).convert('L')
                y_img = Image.open(os.path.join(self.flow_path, record.untrimmed_video_name,
                                                self.image_tmpl[modality].format('y', idx_untrimmed))).convert('L')
            except FileNotFoundError:
                
                for i in range(0, 3):
                    found = True
                    try:
                        x_img = Image.open(os.path.join(self.flow_path, record.untrimmed_video_name,
                                                        self.image_tmpl[modality].format('x',
                                                                                         idx_untrimmed + i))).convert(
                            'L')
                        y_img = Image.open(os.path.join(self.flow_path, record.untrimmed_video_name,
                                                        self.image_tmpl[modality].format('y',
                                                                                         idx_untrimmed + i))).convert(
                            'L')
                    except FileNotFoundError:
                        found = False

                    if found:
                        break
            return [x_img, y_img]
        elif modality == 'Event':
            idx_untrimmed = (record.start_frame // self.args.rgb4e) + idx

            try:
                img_npy = np.load(os.path.join(self.event_path, record.untrimmed_video_name,
                                               self.image_tmpl[modality].format(idx_untrimmed))).astype(
                    np.float32)
            except ValueError:
                img_npy = np.load(os.path.join(self.event_path, record.untrimmed_video_name,
                                               self.image_tmpl[modality].format(
                                                   record.num_frames["Event"]))).astype(
                    np.float32)
            return np.stack([img_npy], axis=0)
        else:
            spec = self._extract_sound_feature(record, idx)
            return [Image.fromarray(spec)]

    def __len__(self):
        return len(self.video_list)
