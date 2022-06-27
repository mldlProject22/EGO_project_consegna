from torch import nn
from utils.transforms import *
from utils.transforms_event import *

def load(double_BN, path):
    state_dict = torch.load(path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k  # [7:]  # remove `module.`
        check_bn = name.split(".")

        if "bn" in check_bn and double_BN:
            print(" * Adapted weight for BN target")
            new_name = name.replace("bn", "bn_target")
            new_state_dict[new_name] = v
            new_state_dict[name] = v
        elif "logits" in check_bn:
            print(" * Skipping Logits weight for \'{}\'".format(name))
            pass
        else:
            # print(" * Param", name)
            new_state_dict[name] = v

        # load params
    return new_state_dict

class I3D(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='bninception', args=None):
        super(I3D, self).__init__()
        self.num_class = num_class
        self.modality = modality
        self.base_arch = base_model
        self.args = args
        self.double_BN = args.double_BN
        self.channels_event = args.channels_events
        self.num_frames_per_clip = num_segments
        self.name = "i3d"
        if self.args.norm_layer:
            self.bn_feat = nn.BatchNorm1d(1024)
        # with this method in self.base_model there is the network chosen
        self._prepare_base_model(base_model, self.channels_event)

    def forward(self, input, is_target):
        # in i3d we implement (as done by authors) the last FC with the Unit3D
        if self.base_arch == 'bninception':
            logits, features, feat_for_cam, weight_softmax = self.base_model(input, self.num_frames_per_clip,
                                                                             is_target, is_event=(self.modality == 'Event'))
        else:
            raise NotImplementedError

        return logits, features, feat_for_cam, weight_softmax

    def _prepare_base_model(self, base_model, channels_events=1):
        print(' * Net architecture: {}\tModality: {}'.format(base_model, self.modality))
        if base_model == "bninception":
            self.feature_dim = 1024
            self.input_size = 224
            # reference https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/mean.py
            self.input_mean = [0.4345, 0.4051, 0.3775]
            self.input_std = [0.2768, 0.2713, 0.2737]
            self.range = [0, 1]

            if self.modality == "RGB":
                weights = load(self.double_BN, self.args.weight_i3d)
                channel = 3
            elif self.modality == 'Flow':
                self.input_mean = [.406]
                self.input_std = [.225]
                path_weight = self.args.weight_i3d_of
                weights = load(self.double_BN, path_weight)
                channel = 2
            elif self.modality == 'Event':
                # Normalization to be tested, not used at the moment
                channel = channels_events
                self.input_mean = np.resize(self.input_mean, channel)
                self.input_std = np.resize(self.input_std, channel)
                weights = load(self.double_BN, self.args.weight_i3d)
                weights_conv1 = weights.pop("Conv3d_1a_7x7.conv3d.weight")
                weight_size = list(weights_conv1.shape)
                weights["Conv3d_1a_7x7.conv3d.weight"] = torch.from_numpy(np.resize(weights_conv1, (weight_size[0],
                                                                                                    1 if self.args.ego3d else channel,
                                                                                                    weight_size[2],
                                                                                                    weight_size[3],
                                                                                                    weight_size[
                                                                                                        4])))

            from archs.i3d_bninception import InceptionI3d
            i3d = InceptionI3d(num_classes=self.num_class,
                               in_channels=1 if self.args.ego3d and self.modality == "Event" else channel,
                               dropout_keep_prob=self.args.dropout,
                               double_BN=self.double_BN, args=self.args)
            i3d.load_state_dict(weights, strict=False)
            self.base_model = i3d

        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        scale_size = {k: v * 256 // 224 for k, v in self.input_size.items()}
        return scale_size

    def get_augmentation(self, modality):
        augmentation = {}
        if modality == 'RGB':
            augmentation = torchvision.transforms.Compose(
                # Data augmentation, at first reduce then interpolate
                [GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                 GroupRandomHorizontalFlip(is_flow=False)])
        if modality == 'Flow':
            augmentation = torchvision.transforms.Compose(
                [GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                 GroupRandomHorizontalFlip(is_flow=True)])
        if modality == 'Event':
            augmentation = ComposeEvents(
                [GroupMultiScaleCropEvents(self.input_size, [1, .875, .75]),
                 GroupRandomHorizontalFlipEvents()])

        return augmentation
