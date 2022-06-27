import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="PyTorch implementation of E2GOMOTION")

parser.add_argument('--modality', type=str, nargs='+', choices=['RGB', 'Flow', 'Event','Spec'],
                    default=['RGB', 'Flow', 'Spec'], help="list of modalities to be used")
parser.add_argument('--train_list', type=str, help="path to list of training samples (by default the one corresponding "
                                                   "to the training split under train_val folder is adopted)")
parser.add_argument('--val_list', type=str, help="path to list of validation samples (by default the one corresponding "
                                                 "to the validation split under train_val folder is adopted)")

parser.add_argument('--visual_path', type=str, default="", help="path to folder with RGB and flow samples")
parser.add_argument('--flow_path', type=str, default="", help="path to folder with flow samples with pwcnet")
parser.add_argument('--event_path', type=str, default="", help="path to voxelgrid samples")

# ========================= Model Configs ==========================
parser.add_argument('--model', type=str, nargs='+', default=['i3d', 'i3d'], help="list of models corresponding to "
                                                                                 "each modality")
parser.add_argument('--base_arch', type=str, nargs='+', default=['bninception', 'bninception'], help="list of backbones"
                                                                                                     " corresponding to"
                                                                                                     " each model")

# Num frames per segment is referred to the number of frames of the relative modality
parser.add_argument('--num_frames_per_clip_train', type=int, nargs='+',
                    help='number of frames per training segment correponding to each modality')
parser.add_argument('--num_frames_per_clip_test', type=int, nargs='+',
                    help='number of frames per testing segment correponding to each modality')
parser.add_argument('--num_clips_test', type=int, default=5,
                    help='number of clips per sample at test time')

parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'],
                    help="type of consensus among segments")
parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')

# ========================= Learning Configs ==========================
# number of iterations I want to do with the batch size I want to simulate
parser.add_argument('--num_iter', default=5000, type=int, metavar='N',
                    help='number of total iter to run (default: 5000)')
parser.add_argument('--resampling_rate', type=int, default=24000)

# actual batch size used in the training
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='partial-batch size (default: 32)')

# batch size to simulate
parser.add_argument('-tb', '--total_batch', default=128, type=int,
                    metavar='N', help='total batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial lr (default: 0.01)')
parser.add_argument('--lr_steps', default=3000, type=int,
                    metavar='LRSteps', help='epochs to decay lr to 0.001')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for optimizer')
parser.add_argument('--weight-decay', '--wd', default=1e-7, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping value (default: None)')

# ========================= Monitor Configs ==========================
parser.add_argument('--eval-freq', '-ef', default=50, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--verbose', '-v', action='store_true',
                    help='Increase verbosity')

# ========================= Runtime Configs ==========================
parser.add_argument('--shift', type=str, default='D1-D1',
                    help='Shift. Format \'source-target\'')
parser.add_argument('--name', default="", type=str,
                    help='name of the experiment')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--gpus', type=str, default=None,
                    help='GPU visible for training (ex. 0,1,2,3) (default: None, thus using all GPUs)')
parser.add_argument('--flow_prefix', default="", type=str,
                    help='the prefix used in the flow files (default: \'\')')
parser.add_argument('--resume_from', type=str, default=None,
                    help='Resume training from a given path')
parser.add_argument('--last', action='store_true',
                    help='resume from last folder (just for test)')

parser.add_argument('--double_BN', action='store_true',
                    help='active double BN, hence separate BN layers for source and target data')
parser.add_argument('--norm_layer', action='store_true', help='normalize features')

# ========================= Event Configs ==========================
parser.add_argument('--rgb4e', type=int, default=6, help='number of rgb frames used for each voxel generated '
                                                         '(to get correspondence)')
parser.add_argument('--normalize_events', action='store_true',
                    help='normalize events with mean and std of pretrained net')
parser.add_argument('--normalize_images', action='store_true',
                    help='normalize images (RGB/Flow) with mean and std of pretrained net')

parser.add_argument('--channels_events', type=int, help='number of channels of the voxelgrid representation', default=9)
parser.add_argument('--resume_from_iteration', type=str, default=None,
                    help='Resume training from a given path (to be used coupled with --iteration)')
parser.add_argument('--iteration', type=str, default=str(3000), help='starting iteration for resume')
parser.add_argument('--dense_sampling_train', type=str2bool, nargs='+', default=[True],
                    help='Dense sampling in train (True/False) for each corresponding modality')
parser.add_argument('--dense_sampling_test', type=str2bool, nargs='+', default=[True],
                    help='Dense sampling in test (True/False) for each corresponding modality')

parser.add_argument('--ego3d', action='store_true', help="Flag to use ego3d")
parser.add_argument('--ego2d', action='store_true', help="Flag to use ego2d")
parser.add_argument('--egomo', default=0, type=float, help="Value of egomo loss")
parser.add_argument('--weight_rna', default=0.0, type=float,
                    help='weight for the AFN loss general ')
parser.add_argument('--weight_gnt', default=1.0, type=float,
                    help='weight for the AFN loss general ')
parser.add_argument('--sync', action='store_true', help="Flag to use ego3d")

parser.add_argument('--rna', action='store_true',help='')

parser.add_argument('--egomo_path_flow', type=str, default="", help="path to voxelgrid samples")

# pwc
parser.add_argument('--pwc', action='store_true', help="if you want to use the flow computed with "
                                                       "pwcnet (in this case flow_pwc_path must be specified)")

########################### NEW KD
parser.add_argument('--egomo_cossim', default=0, type=float, help="Value of egomo_cossim loss")
parser.add_argument('--egomo_feat_patch', default=0, type=float, help="Value of egomo_cossim loss")
parser.add_argument('--egomo_w', default=0, type=float, help="Value of egomo_cossim loss")
parser.add_argument('--flow_classifier_init', action='store_true', help="Start with flow model init")

########################### New RNA-Spatial

parser.add_argument('--rna_spatial', action='store_true',help='')
parser.add_argument('--rna_spatial2', action='store_true',help='')
parser.add_argument('--rna_rgb', action='store_true',help='')
parser.add_argument('--rna_group', action='store_true',help='')
parser.add_argument('--rna_soft', action='store_true',help='')
parser.add_argument('--radius', default=1, type=float, help="radius of RNA")

########################## UDA - Multi DG

parser.add_argument('--UDA', action='store_true',help='')




