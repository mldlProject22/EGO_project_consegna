import os
from functools import reduce
from .transforms import *
from .transforms_event import *
from models.task import Task
from models.I3D import I3D
from models.TSN import TSN
import platform

def take_path(args):
    print(platform.node())
    split = args.shift.split("-")

    # Standard training
    pkl_source = split[0] + "_train.pkl"
    args.train_list = "./train_val/" + pkl_source
    pkl_target = split[-1] + "_test.pkl"  # VAL
    args.val_list = "./train_val/" + pkl_target

    # MULTI SOURCES DG
    if len(split) == 3: #check if I pass more then two sources ex: D1-D2-D3
        print("MULTI DG")
        args.DG = True
        pkl_source2 = split[1] + "_train.pkl"
    else:
        args.DG = False
        pkl_source2 = None


    # UDA
    if args.UDA:
        print("UDA")
        pkl_target_train = split[-1] + "_train.pkl"  # VAL
    else:
        pkl_target_train = None

    print("SOURCE --> ", args.train_list)
    print("VAL --> ", args.val_list)

    ##### UDA --> source - target
    ##### DG --> source - source - test_target

    args.weight_i3d = './pretrained_i3d/rgb_imagenet.pt'
    args.weight_i3d_of = './pretrained_i3d/flow_imagenet.pt'

    if platform.node() == "***": # name machine or os.environ["HOME"].split("/")[-1] == "nomeaccount"
        args.visual_path = "/../rgb_flow"
        args.flow_path = "/../rgb_flow"
        args.event_path = "/../voxels_xy_"+str(args.channels_events)
        args.flow_pwc_path = "/../voxels_xy_"+str(args.channels_events)
        args.weight_i3d = '/../RNA-Relative-Norm-Alignment/pretrained_i3d/rgb_imagenet.pt'
        args.weight_i3d_of = '/../RNA-Relative-Norm-Alignment/pretrained_i3d/flow_imagenet.pt'

        pkl_source = args.shift.split("-")[0] + "_train.pkl"
        pkl_target = args.shift.split("-")[-1] + "_test.pkl"  # VAL
        pkl_target_train = args.shift.split("-")[-1] + "_train.pkl"  # TARGET

        args.train_list = "/../RNA-Relative-Norm-Alignment/train_val/" + pkl_source
        args.val_list = "/../RNA-Relative-Norm-Alignment/train_val/" + pkl_target
        args.train_list_target = "/../RNA-Relative-Norm-Alignment/train_val/" + pkl_target_train
        args.audio_path_model = "/../RNA-Relative-Norm-Alignment/tf_model_zoo/bninception/bn_inception.yaml"
        print(args.train_list_target)

        args.audio_path = "/../RNA-Relative-Norm-Alignment/audio/audio_dic"
        args.audio_path_model = "/../RNA-Relative-Norm-Alignment/tf_model_zoo/bninception/bn_inception.yaml"

    else:
        #args.visual_path = "/data/EpicKitchenDA/rgb_flow"
        #args.flow_path = "/data/EpicKitchenDA/rgb_flow"
        #args.event_path = "/data/EK55_events/voxels_xy_" + str(args.channels_events)
        #args.flow_pwc_path = "/data/EK55_events/voxels_xy_" + str(args.channels_events)
        # args.train_list = "./train_val/D1_train.pkl"
        # args.val_list = "./train_val/D1_test.pkl"
        #args.weight_i3d = '/home/mirco/ActivityRecognition_DA/pretrained_i3d/rgb_imagenet.pt'
        #args.weight_i3d_of = '/home/mirco/ActivityRecognition_DA/pretrained_i3d/flow_imagenet.pt'

        pkl_source = args.shift.split("-")[0] + "_train.pkl"
        pkl_target = args.shift.split("-")[-1] + "_test.pkl"  # VAL
        pkl_target_train = args.shift.split("-")[-1] + "_train.pkl"  # TARGET

        #args.train_list = "/home/mirco/ActivityRecognition_DA/train_val/" + pkl_source
        #args.val_list = "/home/mirco/ActivityRecognition_DA/train_val/" + pkl_target
        #args.train_list_target = "/home/mirco/ActivityRecognition_DA/train_val/" + pkl_target_train
        #args.audio_path_model = "/home/chiara/RNA_clean/tf_model_zoo/bninception/bn_inception.yaml"
        #print(args.train_list_target)

        #args.audio_path = "/home/mirco/AUDIO_EK/audio/audio_dic"
        #args.audio_path_model = "/home/chiara/RNA_clean/tf_model_zoo/bninception/bn_inception.yaml"
        if args.UDA:
            #args.train_list_target = "/home/mirco/ActivityRecognition_DA/train_val/" + pkl_target_train
            args.train_list2 = None
        if args.DG:
            args.train_list_target = None
            #args.train_list2 = "/home/mirco/ActivityRecognition_DA/train_val/" + pkl_source2
            print("SOURCE 2--> ", args.train_list2)
    '''
        args.visual_path = "/../EpicKitchenVoxel/rgb_flow/"
        args.flow_path = "/../EpicKitchenVoxel/rgb_flow/"
        args.event_path = '/../EpicKitchenVoxel/voxels_xy_3/'
        args.flow_pwc_path = '/../EpicKitchenVoxel/voxels_xy_3/'

        args.weight_i3d_of = '/../pretrained_i3d/flow_imagenet.pt'
        args.weight_i3d = '/../pretrained_i3d/rgb_imagenet.pt'


        pkl_source = args.shift.split("-")[0] + "_train.pkl"

        pkl_target = args.shift.split("-")[-1] + "_test.pkl"  # VAL
        pkl_target_train = args.shift.split("-")[-1] + "_train.pkl"  # TARGET


        args.train_list = "/../RNA-Relative-Norm-Alignment/train_val/" + pkl_source
        args.val_list = "/../RNA-Relative-Norm-Alignment/train_val/" + pkl_target
        args.train_list_target = "/../RNA-Relative-Norm-Alignment/train_val/" + pkl_target_train
    '''


def set_domain_shift(args):
    domains = {'D1': 8, 'D2': 1, 'D3': 22}
    source = domains[args.shift.split("-")[0]]
    target = domains[args.shift.split("-")[1]]
    target_test = None

    return source, target, target_test


def get_domains_and_labels(args):
    source_domains, target_domains, target_test = set_domain_shift(args)
    valid_domains = {'train': [source_domains, target_domains],
                     'val': [target_domains]}  # selection of the domains at train and validation time
    N_VERBS = 8

    valid_labels = {'verb': [i for i in range(N_VERBS)]}
    num_class = N_VERBS
    return num_class, valid_domains, valid_labels, source_domains, target_domains, target_test


def data_preprocessing(model, modalities, flow_prefix, args):
    crop_size = {m: model[m].crop_size for m in modalities}
    train_augmentation = {m: model[m].get_augmentation(m) for m in modalities}
    image_tmpl = {}
    train_transform = {}
    val_transform = {}
    load_cineca_data = os.environ["HOME"].split("/")[-1] == "abottin1"

    for m in modalities:
        # Prepare dictionaries containing image name templates for each modality
        if m == 'RGB':
            image_tmpl[m] = "frame_{:010d}.jpg" if load_cineca_data else "img_{:010d}.jpg"
        elif m == 'Flow':
            image_tmpl[m] = "frame_{:010d}.jpg" if load_cineca_data else flow_prefix + "{}_{:010d}.jpg"
        elif m == 'Event':
            image_tmpl[m] = "event_{:010d}.npy"
        # Prepare train/val dictionaries containing the transformations
        # (augmentation+normalization)
        # for each modality
        if m == 'Event':
            train_transform[m] = ComposeEvents([
                train_augmentation[m],
                StackEvents(args, model[m].input_mean, model[m].input_std, model[m].range),
                ToTensorEvents()
            ])

            val_transform[m] = ComposeEvents([
                GroupCenterCropEvents(crop_size[m]),
                StackEvents(args, model[m].input_mean, model[m].input_std, model[m].range),
                ToTensorEvents()
            ])
        elif m == 'Spec':
            train_transform[m] = torchvision.transforms.Compose([
                Stack(roll=True),
                ToTorchFormatTensor(div=False),
            ])

            val_transform[m] = torchvision.transforms.Compose([
                Stack(roll=True),
                ToTorchFormatTensor(div=False),
            ])

        else:
            train_transform[m] = torchvision.transforms.Compose([
                train_augmentation[m],
                Stack(roll=False),
                ToTorchFormatTensor(div=not args.normalize_images),
                GroupNormalize(args, model[m].input_mean, model[m].input_std, model[m].range)
            ])

            val_transform[m] = torchvision.transforms.Compose([
                GroupCenterCrop(crop_size[m]),
                Stack(roll=False),
                ToTorchFormatTensor(div=not args.normalize_images),
                GroupNormalize(args, model[m].input_mean, model[m].input_std, model[m].range)
            ])
            print(val_transform)

    return image_tmpl, train_transform, val_transform

def load(weight):
    new_weight = {}

    checkpoint = torch.load(weight)

    iter = checkpoint['iteration']
    best_iter = checkpoint['best_iter']
    best_iter_score = checkpoint['best_iter_score']
    acc_mean = checkpoint['acc_mean']
    loss_mean = checkpoint['loss_mean']
    
    for i in checkpoint["model_state_dict"].keys():

        if "module.base_model.conv1" in i:
            pass
        else:
            new_weight[i] = checkpoint["model_state_dict"][i]

    checkpoint["model_state_dict"] = new_weight
    return checkpoint

def load_checkpoint(path, model, optimizer=None):
    print(path)
    checkpoint = torch.load(path)
    #checkpoint = torch.load(path,map_location=torch.device('cpu')) #use this instead of previous row in case of CPU
    iter = checkpoint['iteration']
    best_iter = checkpoint['best_iter']
    best_iter_score = checkpoint['best_iter_score']
    acc_mean = checkpoint['acc_mean']
    loss_mean = checkpoint['loss_mean']
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    device = torch.device('cuda')
    #device = torch.device('cpu') #use this instead of previous row in case of CPU
    model.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, iter, best_iter, best_iter_score, acc_mean, loss_mean

def load_checkpoint_flow(path, model, optimizer=None, args=None):
    print(path)
    if args is not None:
        checkpoint = load(path) if args.flow_classifier_init else torch.load(path)
    else:
        checkpoint = torch.load(path)
    iter = checkpoint['iteration']
    best_iter = checkpoint['best_iter']
    best_iter_score = checkpoint['best_iter_score']
    acc_mean = checkpoint['acc_mean']
    loss_mean = checkpoint['loss_mean']
    model.load_state_dict(checkpoint['model_state_dict'], strict=False) 
    device = torch.device('cuda')
    model.to(device)

    return model, None, iter, best_iter, best_iter_score, acc_mean, loss_mean


def save_checkpoint(model, optimizer, iteration, best_iter, best_accuracy, acc_mean, loss_mean,
                    experiment_dir, filename):
    weights_dir = os.path.join('saved_models', experiment_dir)
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    try:
        torch.save({'iteration': iteration,
                    'best_iter': best_iter,
                    'best_iter_score': best_accuracy,
                    'acc_mean': acc_mean,
                    'loss_mean': loss_mean,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, os.path.join(weights_dir, filename))

    except Exception as e:
        print("An error occurred while saving the checkpoint:")
        print(e)


total_params = 0


def init_model(networks, name, args, summaryWriter):
    # in case we have a model indipendent from the modality (i.e. ss task) add a key to maintain coherence
    if not isinstance(networks, dict):
        networks = {'Mixed': networks}

    if name == "action-classifier":
        criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                              reduce=None, reduction='none')
    else:
        criterion = torch.nn.CrossEntropyLoss()

    learning_rate = args.lr
    param_groups = {modality: filter(lambda p: p.requires_grad, model.parameters()) for modality, model in
                    networks.items()}
    n_params = 0
    for modality, model in networks.items():
        for _, p in model.named_parameters():
            if p.requires_grad:
                n_params += p.numel()

    global total_params
    total_params += n_params
    print('TOTAL PARAMS ' + name + ' = ' + str(n_params))
    print('TOTAL PARAMS (until now) = ' + str(total_params))

    # SGD optimizer for all networks
    optimizer = {modality: torch.optim.SGD(param_groups[modality], learning_rate, weight_decay=args.weight_decay,
                                           momentum=args.momentum)
                 for modality in networks.keys()}

    task = Task(name, networks, criterion, optimizer, args.batch_size, args.total_batch, summaryWriter, args=args)
    return task

'''
    Calculate accuracy
'''


def compute_accuracy(output, label, batch_size, top1, top5, verb_top1, verb_top5, noun_top1, noun_top5, summaryWriter,
                     iter):
    LOG_FREQUENCY = 50
    verb_only = len(list(output.values())[0].shape) == 2

    # Apply softmax, to calculate accuracy
    # output = {m: nn.Softmax(dim=1)(out) for m, out in output.items()}
    # Fuse outputs from different modalities
    out_verb_fused = reduce(lambda x, y: x + y if verb_only else x[0] + y[0],
                            output.values())
    # Compute accuracy
    verb_prec1, verb_prec5, class_correct, class_total = accuracy(out_verb_fused, label['verb'],
                                                                  topk=(1, 5))
    # Update values
    verb_top1.update(verb_prec1, batch_size, class_correct, class_total)
    verb_top5.update(verb_prec5, batch_size, class_correct, class_total)

    if iter % LOG_FREQUENCY == 0 and summaryWriter != None:
        summaryWriter.add_scalar("test_top1", verb_top1.val, iter + 1)

    prec1, prec5 = verb_prec1, verb_prec5
    top1.update(prec1, batch_size, class_correct, class_total)
    top5.update(prec5, batch_size, class_correct, class_total)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.correct = list(0. for _ in range(8))
        self.total = list(0. for _ in range(8))

    def update(self, val, n=1, class_correct=None, class_total=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if class_correct is not None and class_total is not None:
            for i in range(0, 8):
                self.correct[i] += class_correct[i]
                self.total[i] += class_total[i]

    def __get__(self, instance, owner):
        return self.val

    def __add__(self, other):
        tot_val = self.val + other.val
        # sum = AverageMeter()
        # sum.update(tot_val, self.count)
        return tot_val


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    num_label = output.size(1)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    class_correct, class_total = accuracy_per_class(correct[:1].view(-1), target, num_label)
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).to(torch.float32).sum(0)
        res.append(float(correct_k.mul_(100.0 / batch_size)))
    if len(res) == 1:
        res.append(0)
    return res[0], res[1], class_correct, class_total


def loss(criterion, logits, label):
    # Calculate classification loss L_y
    # perform sum of the output (activations) of each modality
    loss_verb = criterion(logits, label['verb'])
    return loss_verb


def accuracy_per_class(correct, target, num_label):
    class_correct = list(0. for _ in range(0, num_label))
    class_total = list(0. for _ in range(0, num_label))
    for i in range(0, target.size(0)):
        class_label = target[i].item()
        class_correct[class_label] += correct[i].item()
        class_total[class_label] += 1
    return class_correct, class_total


def multitask_accuracy(outputs, labels, topk=(1,)):
    """
    Args:
        outputs: tuple(torch.FloatTensor), each tensor should be of shape
            [batch_size, class_count], class_count can vary on a per task basis, i.e.
            outputs[i].shape[1] can be different to outputs[j].shape[j].
        labels: tuple(torch.LongTensor), each tensor should be of shape [batch_size]
        topk: tuple(int), compute accuracy at top-k for the values of k specified
            in this parameter.
    Returns:
        tuple(float), same length at topk with the corresponding accuracy@k in.
    """
    max_k = int(np.max(topk))
    task_count = len(outputs)
    batch_size = labels[0].size(0)
    all_correct = torch.zeros(max_k, batch_size).type(torch.ByteTensor)
    if torch.cuda.is_available():
        all_correct = all_correct.cuda()
    for output, label in zip(outputs, labels):
        _, max_k_idx = output.topk(max_k, dim=1, largest=True, sorted=True)
        # Flip batch_size, class_count as .view doesn't work on non-contiguous
        max_k_idx = max_k_idx.t()
        correct_for_task = max_k_idx.eq(label.view(1, -1).expand_as(max_k_idx))
        all_correct.add_(correct_for_task)

    accuracies = []
    for k in topk:
        all_tasks_correct = torch.ge(all_correct[:k].float().sum(0), task_count)
        accuracy_at_k = float(all_tasks_correct.float().sum(0) * 100.0 / batch_size)
        accuracies.append(accuracy_at_k)
    return tuple(accuracies)


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]

def get_model_spec(model, args, m):
    net = {"i3d": I3D, "TSN": TSN, "TSM": TSN}

    """
    egments are just for TSM, TSN; test segments are set to 1 for TSN since the testing of this network works 
    differently (according to the original repo); while TSM averages the predictions of each segments with the 
    ConsensusModule within the forward function, TSN just outputs the prediction of one segment and the final output 
    is obtained averaging all the outputs outside the forward (look at validate for more details)
    """
    num_segments_test = {"i3d": args.num_frames_per_clip_test[args.modality.index(m)],
                         "TSN": 1, "TSM": args.num_frames_per_clip_test[args.modality.index(m)]}

    # num_frames per clip in training
    num_frames_per_clip_train = {"i3d": args.num_frames_per_clip_train[args.modality.index(m)],
                                 "TSN": args.num_frames_per_clip_train[args.modality.index(m)],
                                 "TSM": args.num_frames_per_clip_train[args.modality.index(m)]}

    # num_frames per clip in test and validation
    num_frames_per_clip_test = {"i3d": args.num_frames_per_clip_test[args.modality.index(m)],
                                "TSN": args.num_frames_per_clip_test[args.modality.index(m)],
                                "TSM": args.num_frames_per_clip_test[args.modality.index(m)]}

    kwargs = {"i3d": {}, "TSN": {"is_shift": False},
              "TSM": {"is_shift": True}}

    return {"net": net[model], "segments_test": num_segments_test[model],
            "num_frames_per_clip_train": num_frames_per_clip_train[model],
            "num_frames_per_clip_test": num_frames_per_clip_test[model],
            "kwargs": kwargs[model]}


def sync_train_test_model(train_models, test_models):
    if not isinstance(train_models, list):
        model_list_train = [train_models]
    if not isinstance(test_models, list):
        model_list_test = [test_models]
    modalities = model_list_test[0].keys()
    for i, model in enumerate(model_list_test):
        for m in modalities:
            print("Sync weights for modality {} model {}".format(m, i))
            model[m].load_state_dict(model_list_train[i][m].state_dict())