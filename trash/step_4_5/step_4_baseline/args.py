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


parser = argparse.ArgumentParser(description="STEP 3")

parser.add_argument('--modality', type=str,   help="list of modalities to be used (RGB or Flow)")

parser.add_argument('--model', type=str, help="list of models corresponding to each modality (i3d or TSM)")

parser.add_argument('--shift', type=str,   help='Shift. Format \'source-target\'')

parser.add_argument('--n_classes', type=int, default = 8 ,  help='Number of target classes, default 8')

