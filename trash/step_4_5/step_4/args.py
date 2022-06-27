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


parser = argparse.ArgumentParser(description="STEP 4")

parser.add_argument('--shift', type=str, default="D1-D2" , help='Shift. Format \'source-target\'')

parser.add_argument('--aggregation', type=str)

parser.add_argument('--n_classes', type=int, default = 8 ,  help='Number of target classes, default 8')

parser.add_argument('--place_adv', default=['Y', 'Y', 'Y'], type=str, nargs="+",
                    metavar='N', help='[video relation-based adv solo trn-m, video-based adv, frame-based adv]')

parser.add_argument('--batch_size', type=int, default = 128,  help='batch size validation')

parser.add_argument('--use_attn', type=str, default = 'none', choices = ['none', 'TransAttn', 'general'] ,help='type of attention')

parser.add_argument('--epochs', type=int, default = 20 ,  help='Number of epochs')

parser.add_argument('--trials', type=int, default = 20 ,  help='Number of trials')

parser.add_argument('--lr', type=float, default = 3e-3 ,  help='learning rate')

parser.add_argument('--beta', type=float, default = [0.75, 0.75, 0.75] , nargs="+",  help='beta')

parser.add_argument('--gamma', type=float, default = 0.003 ,  help='gamma')

parser.add_argument('--modality', type=str,
                    help='modality')

parser.add_argument('--loss_weights', type=float, default=[0.75, 0.5, 0.75],nargs="+",
                    help='losses')                   



#parser.add_argument('--place_adv', type=str, default = "Y Y Y")
