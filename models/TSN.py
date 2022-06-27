# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

from torch import nn

import archs.resnet50
from archs.consensus import ConsensusModule
from utils.transforms import *
from utils.transforms_event import *
from torch.nn.init import normal_, constant_

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet50',
                 args=None, **kwargs):

        super(TSN, self).__init__()
        self.modality = modality
        self.args = args
        self.num_segments = num_segments
        self.channels_event = args.channels_events
        self.reshape = True
        self.before_softmax = kwargs.get("before_softmax", True)
        self.dropout = args.dropout
        self.crop_num = kwargs.get("crop_num", 1)
        self.consensus_type = args.consensus_type
        self.img_feature_dim = kwargs.get("img_feature_dim", 256)  # the dimension of the CNN feature to represent each frame
        self.pretrain = kwargs.get("pretrain", 'imagenet')

        self.is_shift = kwargs.get("is_shift", False)
        self.shift_div = kwargs.get("shift_div", 8)
        self.shift_place = kwargs.get("shift_place", 'blockres')
        self.base_arch = base_model
        self.fc_lr5 = kwargs.get("fc_lr5", False)
        self.temporal_pool = kwargs.get("temporal_pool", False)
        self.non_local = kwargs.get("non_local", False)
        self.name = "TSM" if self.is_shift else "TSN"

        if not self.before_softmax and self.consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if kwargs.get("print_spec", True):
            print(("""
    Initializing {} with base model: {}.
    {} Configurations:
        input_modality:     {}
        num_segments:       {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(self.name, self.name, base_model, self.modality, self.num_segments, self.consensus_type, self.dropout,
                       self.img_feature_dim)))

        self._prepare_base_model(base_model)

        self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        if self.modality == 'Event':
            print("Converting the ImageNet model to a event init model")
            if self.args.channels_events != 3:
                self.base_model = self._construct_event_model(self.base_model)
            print("Done. Event model ready...")
        elif self.modality == "Spec":
            print(" * Converting the ImageNet model to a spectrogram init model")
            if self.base_arch == "bninception":
                self.base_model = self._construct_spec_model(self.base_model)

        self.consensus = ConsensusModule(self.consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = kwargs.get("partial_bn", False)
        if self._enable_pbn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            if hasattr(self.new_fc, 'weight'):
                normal_(self.new_fc.weight, 0, std)
                constant_(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):
        print('=> base model: {}'.format(base_model))

        if base_model == 'resnet50':
            self.base_model = archs.resnet50.resnet50(pretrained=True,
                                                      ego2d=(self.args.ego2d and self.modality == 'Event'))
            if self.is_shift:
                print('Adding temporal shift...')
                from ops.temporal_shift import make_temporal_shift
                make_temporal_shift(self.base_model, self.num_segments,
                                    n_div=self.shift_div, place=self.shift_place, temporal_pool=self.temporal_pool)

            if self.non_local:
                print('Adding non-local module...')
                from ops.non_local import make_non_local
                make_non_local(self.base_model, self.num_segments)

            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            self.range = [0, 1]
            self.base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'Event':
                self.input_mean = np.resize(self.input_mean, self.args.channels_events)
                self.input_std = np.resize(self.input_std, self.args.channels_events)


        elif base_model == 'bninception':

            import tf_model_zoo
            base_model = 'BNInception'
            self.base_model = getattr(tf_model_zoo, base_model)(model_path=self.args.audio_path_model)
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]

            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

            elif self.modality == 'RGB':
                self.input_mean = [104, 117, 128]

            self.feature_dim = 1024


            if self.modality == 'Flow':
                self.input_mean = [128]

            elif self.modality == 'Event':
                self.input_mean = np.resize(self.input_mean, self.args.channels_events)
                self.input_std = np.resize(self.input_std, self.args.channels_events)

            else:
                self.input_size = 224
                self.input_std = [1]

            if self.is_shift:
                print('Adding temporal shift...')
                self.base_model.build_temporal_ops(
                    self.num_segments, is_temporal_shift=self.shift_place, shift_div=self.shift_div)

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn and mode:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()
                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                if self.fc_lr5:
                    lr5_weight.append(ps[0])
                else:
                    normal_weight.append(ps[0])
                if len(ps) == 2:
                    if self.fc_lr5:
                        lr10_bias.append(ps[1])
                    else:
                        normal_bias.append(ps[1])

            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"},
        ]

    def forward(self, input, is_target, no_reshape=False):
        if not no_reshape:
            channels = {"RGB": 3, "Event": self.channels_event, "Flow": 2, "Spec":1}
            sample_len = channels[self.modality]
            base_out = self.base_model(input.contiguous().view((-1, sample_len) + input.size()[-2:]))
        else:
            base_out = self.base_model(input)
        spatial_feat = base_out
        spatial_feat = spatial_feat.view((-1, self.num_segments) + spatial_feat.size()[1:])
        if self.base_arch == 'resnet50':
            base_out = self.base_model.avgpool(base_out)
            base_out = torch.flatten(base_out, 1)

        feat = base_out
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
        if not self.before_softmax:
            base_out = self.softmax(base_out)

        if self.reshape:
            if self.is_shift and self.temporal_pool:
                base_out = base_out.view((-1, self.num_segments // 2) + base_out.size()[1:])

            else:
                base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            output = self.consensus(base_out)
            return output.squeeze(1), feat, spatial_feat, None

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        if self.base_arch == 'bninception':
            import torch.utils.model_zoo as model_zoo
            sd = model_zoo.load_url('https://www.dropbox.com/s/35ftw2t4mxxgjae/BNInceptionFlow-ef652051.pth.tar?dl=1')
            for layer in list(sd.keys()):
                if "conv1" in layer:
                    del sd[layer]
            base_model.load_state_dict(sd, strict=False)
            print('=> Loading pretrained Flow weight done...')
        else:
            print('#' * 30, 'Warning! No Flow pretrained model is found')
        return base_model


    def _construct_spec_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        new_kernels = params[0].detach().mean(dim=1, keepdim=True).contiguous()

        new_conv = nn.Conv2d(1, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].detach()  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)

        #replace the avg pooling at the end, so that it matches the spectrogram dimensionality (256x256)
        pool_layer = getattr(self.base_model, 'global_pool')
        new_avg_pooling = nn.AvgPool2d(8, stride=pool_layer.stride, padding=pool_layer.padding)
        setattr(self.base_model, 'global_pool', new_avg_pooling)

        return base_model


    def _construct_event_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (self.channels_event,) + kernel_size[2:]
        new_kernels = torch.from_numpy(np.resize(params[0].data, new_kernel_size)).contiguous()
        new_conv = nn.Conv2d(self.channels_event, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)

        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self, flip=True):
        if self.modality == 'RGB':
            if flip:
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                       GroupRandomHorizontalFlip(is_flow=False)])
            else:
                print('#' * 20, 'NO FLIP!!!')
                return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'Event':
            return ComposeEvents([GroupMultiScaleCropEvents(self.input_size, [1, .875, .75]),
                                  GroupRandomHorizontalFlipEvents()])
