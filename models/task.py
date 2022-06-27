import torch
import os
from pathlib import Path
from utils import utils
from functools import reduce
import random
import torch.nn.functional as F

LOG_FREQUENCY = 1


def get_L2norm_loss_self_driven(x, radius, modality=None):
    if modality is not None:
        print("norm of " + modality + " --> " + str(x.norm(p=2, dim=1).mean()))
    else:
        print("Norm of fraction RGB/FLOW ", x)
    # radius = radius + 0.1
    loss = (x - radius) ** 2
    return loss
    
def get_L2norm_loss_self_driven_spatial(x, radius, modality=None):
    # radius = radius + 0.1
    loss = (x - radius) ** 2
    return loss

def get_L2norm_loss_self_driven_soft(x, radius):
    #radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 0.01
    loss = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return loss


def COT_unsupervised(pred, label, classes=8):
    pred_soft = F.softmax(pred, dim=1)
    _, label = torch.max(pred_soft, 1)
    batch_size = len(label)
    Yg = torch.gather(pred_soft, 1, torch.unsqueeze(label, 1))
    Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
    Px = pred_soft / Yg_.view(len(pred_soft), 1)
    Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
    y_zerohot = torch.ones(batch_size, classes).scatter_(1, label.view(batch_size, 1).data.cpu(), 0)
    output = Px * Px_log * y_zerohot.cuda()
    loss = torch.sum(output)
    loss /= float(batch_size)
    loss /= float(classes)
    return loss


def COT(output, label, classes=8):
    #
    # “COMPLEMENT OBJECTIVE TRAINING” ICLR19
    # https://github.com/henry8527/COT/blob/master/code/COT.py
    #
    batch_size = len(label)
    output = F.softmax(output, dim=1)
    Yg = torch.gather(output, 1, torch.unsqueeze(label, 1))
    Yg_ = (1 - Yg) + 1e-7  # avoiding numerical issues (first)
    Px = output / Yg_.view(len(output), 1)
    Px_log = torch.log(Px + 1e-10)  # avoiding numerical issues (second)
    y_zerohot = torch.ones(batch_size, classes).scatter_( 1, label.view(batch_size, 1).data.cpu(), 0)
    output = Px * Px_log * y_zerohot.cuda()
    loss = torch.sum(output)
    loss /= float(batch_size)
    loss /= float(classes)
    return loss




class Task:
    def __init__(self, name, models, criterion, optimizer, batch_size, total_batch,
                 summaryWriter, args) -> None:
        super().__init__()
        self.name = name  # name of the task (action classifier, domain classifier, self-super)
        self.models = models  # {'RGB': rgb_model, 'Flow': flow_model}, {'Mixed': ss_task_model}. It's a dict (str, Net)
        # can contain multiple models (one for modality)
        self.criterion = criterion
        self.modalities = list(self.models.keys())
        self.optimizer = optimizer
        # Other useful parameters
        self.batch_size = batch_size
        self.total_batch = total_batch
        self.summaryWriter = summaryWriter
        self.args = args
        self.KD_from_flow = (args.egomo > 0) or \
                            (args.egomo_cossim > 0) or \
                            (args.egomo_feat_patch > 0) or \
                            (args.egomo_w > 0)  # add here others KD version


    def load_on_GPU(self, device=torch.device('cuda')):
        for modality, model in self.models.items():
            self.models[modality] = torch.nn.DataParallel(model).to(device)

    def resume_model(self, path):
        print('Restoring ---> {}'.format(self.name))
        # list all files in chronological order (1st is most recent, last is less recent)
        saved_models = [x for x in reversed(sorted(Path(path).iterdir(), key=os.path.getmtime))]
        iter, best_iter, best_iter_score, acc_mean, loss_mean = 0, 0, 0, 0, 0
        for m in self.modalities:
            # get only models which belong to this task and for this modality
            modality_models = list(filter(lambda x: m == x.name.split('.')[0].split('_')[-2]  # modality check
                                                    and self.name == x.name.split('.')[0].split('_')[-3],
                                          # model name check
                                          saved_models))

            # get the most recently saved one
            model_path = os.path.join(path, modality_models[0].name)
            self.models[m], self.optimizer[m], iter, best_iter, best_iter_score, acc_mean, loss_mean = \
                utils.load_checkpoint(model_path, self.models[m], self.optimizer[m])
            print(' * Mode: {} OK! File = \'{}\''.format(m, modality_models[0].name))

        iter += 1
        if isinstance(self, ActionRecognition):
            self.verb_top1.update(acc_mean, self.batch_size * iter)
            self.loss_verb.update(loss_mean, self.batch_size * iter)
            self.best_iter = best_iter
            self.best_iter_score = best_iter_score

        return iter

    def resume_flow_model(self, path):
        print('Restoring Flow Model ---> {}'.format(self.name))
        # list all files in chronological order (1st is most recent, last is less recent)
        # saved_models = [x for x in reversed(sorted(Path(path).iterdir(), key=os.path.getmtime))]
        iter, best_iter, best_iter_score, acc_mean, loss_mean = 0, 0, 0, 0, 0
        m = "Flow"
        if 'Flow' in self.models.keys():
            self.models[m], _, iter, best_iter, best_iter_score, acc_mean, loss_mean = \
                utils.load_checkpoint_flow(path, self.models[m])
        print(" acc_mean --> ", acc_mean)
        print(" loss --> ", loss_mean)
        print(" best_iter --> ", best_iter)
        print(" best_iter_score --> ", best_iter_score)

    def resume_from_flow_model(self, path):
        print('Restoring Flow Model in Event Model as INIT ---> {}'.format(self.name))
        # list all files in chronological order (1st is most recent, last is less recent)
        # saved_models = [x for x in reversed(sorted(Path(path).iterdir(), key=os.path.getmtime))]
        iter, best_iter, best_iter_score, acc_mean, loss_mean = 0, 0, 0, 0, 0
        m = "Event"
        for m in self.modalities:
            if m != 'Flow':
                self.models[m], _, iter, best_iter, best_iter_score, acc_mean, loss_mean = \
                    utils.load_checkpoint_flow(path, self.models[m], args=self.args)


    def resume_model_iteration(self, path, iteration):
        print('Restoring ---> {}'.format(self.name))
        # list all files in chronological order (1st is most recent, last is less recent)
        saved_models = [x for x in reversed(sorted(Path(path).iterdir(), key=os.path.getmtime))]
        iter, best_iter, best_iter_score, acc_mean, loss_mean = 0, 0, 0, 0, 0
        for m in self.modalities:
            # get only models which belong to this task and for this modality
            modality_models = list(filter(lambda x: m == x.name.split('.')[0].split('_')[-2]  # modality check
                                                    and self.name == x.name.split('.')[0].split('_')[-3]
                                                    and iteration == x.name.split('.')[0].split('_')[-4],
                                          # model name check
                                          saved_models))

            # get the most recently saved one
            model_path = os.path.join(path, modality_models[0].name)
            self.models[m], self.optimizer[m], iter, best_iter, best_iter_score, acc_mean, loss_mean = \
                utils.load_checkpoint(model_path, self.models[m], self.optimizer[m])
            print(' * Mode: {} OK! File = \'{}\''.format(m, modality_models[0].name))

        iter += 1
        if isinstance(self, ActionRecognition):
            self.verb_top1.update(acc_mean, self.batch_size * iter)
            self.loss_verb.update(loss_mean, self.batch_size * iter)
            self.best_iter = best_iter
            self.best_iter_score = best_iter_score

        return iter

    def save_model(self, iter, best_iter, best_iter_score, verb_top1, loss_classification_verb, experiment_dir, count,
                   optimizer, pre):
        for m in self.modalities:
            utils.save_checkpoint(self.models[m], optimizer[m] if type(optimizer) is dict else optimizer, iter,
                                  best_iter, best_iter_score,
                                  verb_top1.avg if (not (type(verb_top1) is dict) and verb_top1 != None) else verb_top1[
                                      m].avg if verb_top1 != None else None,
                                  loss_classification_verb.avg if (
                                          not (type(
                                              loss_classification_verb) is dict) and loss_classification_verb != None) else
                                  loss_classification_verb[m].avg if loss_classification_verb != None else None,
                                  experiment_dir,
                                  filename=pre + '_' + self.name + '_' + m + '_' + str(count) + '.pth')

    def train(self, mode=True):
        # activate the training in all modules (when training, DropOut is active, BatchNorm updates itself)
        # (when not training, BatchNorm is freezed, DropOut disabled)
        for model in self.models.values():
            model.train(mode)
        if self.KD_from_flow:
            if 'Flow' in self.models.keys():
                print('train mode False')
                self.models["Flow"].train(False)

    def zero_grad(self):
        for m in self.modalities:
            if m == 'Flow' and self.KD_from_flow:
                pass
            else:
                self.optimizer[m].zero_grad()

    def step(self):
        for m in self.modalities:
            if m == 'Flow' and self.KD_from_flow:
                pass
            else:
                self.optimizer[m].step()
            # print("lr " + m + " " + str(self.optimizer[m].param_groups[-1]["lr"]) + " " + self.name)

    def __str__(self) -> str:
        return self.name


# CLASSES BELOW EXTEND Task


'''
    Action recognition task
'''


class ActionRecognition(Task):
    def __init__(self, task, source_domain, target_domain) -> None:
        super().__init__(task.name, task.models, task.criterion, task.optimizer, task.batch_size,
                         task.total_batch, task.summaryWriter, task.args)
        # Accuracy measures
        self.KD_from_flow = (task.args.egomo > 0) or \
                            (task.args.egomo_cossim > 0) or \
                            (task.args.egomo_feat_patch > 0) or \
                            (task.args.egomo_w > 0)  
        self.best_iter = 0
        self.best_iter_score = 0.0
        self.top1 = utils.AverageMeter()
        self.top5 = utils.AverageMeter()
        self.verb_top1 = utils.AverageMeter()
        self.verb_top5 = utils.AverageMeter()
        self.noun_top1 = utils.AverageMeter()
        self.noun_top5 = utils.AverageMeter()
        # Losses
        self.loss_verb = utils.AverageMeter()
        self.loss_noun = None
        if self.args.rna:
            self.loss_rna = utils.AverageMeter()
        if self.KD_from_flow:
            self.loss_egomo = utils.AverageMeter()
            self.criterion_egomo = torch.nn.MSELoss()
            self.loss_egomo_cossim = utils.AverageMeter()
            self.criterion_egomo_cossim = torch.nn.CosineSimilarity()
            self.loss_egomo_feat_patch = utils.AverageMeter()
            self.criterion_egomo_feat_patch = torch.nn.MSELoss()
            self.loss_egomo_w = utils.AverageMeter()
            self.criterion_egomo_w = torch.nn.MSELoss(reduce=False)

        self.supervised = source_domain == target_domain

    def execute_task(self, input, label, retain_graph, is_target, weight_class):
        # it does: forward in the model, compute loss and acc, backward

        logits, features, feat_for_cam, weight_softmax = self.forward(input, is_target=is_target)

        if not is_target:
            self.compute_loss(logits, label, classification_weight=weight_class)
        if self.args.egomo > 0:
            self.compute_egomo_loss(features)
        if self.args.egomo_cossim > 0:
            self.compute_egomo_cossim_loss(features)
        if self.args.egomo_feat_patch > 0:
            self.compute_egomo_feat_patch_loss(feat_for_cam)
        if self.args.egomo_w > 0:
            self.compute_egomo_w_loss(features, logits, label)
        if not is_target:
            self.compute_accuracy(logits, label)
            self.backward(retain_graph=retain_graph)
        return logits, features, feat_for_cam, weight_softmax

    def forward(self, input, is_target):
        # it is done for each modality and then it is all saved in dicts
        # Get softmax for each modality for the action classification task and the features
        logits = {}
        features = {}  # useful for the SS task
        weight_softmax = {}
        feat_for_cam = {}
        for m in self.modalities:
            if m == "Flow" and self.KD_from_flow:
                _, features[m], feat_for_cam[m], weight_softmax[m] = self.models[m](input=input[m],
                                                                                    is_target=is_target)
                features[m] = features[m]
            else:
                logits[m], features[m], feat_for_cam[m], weight_softmax[m] = self.models[m](input=input[m],
                                                                                            is_target=is_target)
            # print("Features ", m, features[m].norm(p=2, dim=1).mean())

        # every variable is a dict indexed by the modalities (rgb, flow, event).
        # For each index, their shape is:
        # logits        = (batch size, num_classes)
        # features      = (batch size, 1024)
        # weight_softmax= (batch size, 1024, 1, 1, 1)
        # feat_for_cam  = (batch size, 1024, 2, 7, 7)
        return logits, features, feat_for_cam, weight_softmax

    def compute_loss(self, logits, label, classification_weight=1.):
        # Calculate classification loss L_y
        # perform sum of the output (activations) of each modality
        out_verb_fused = reduce(lambda x, y: x + y,
                                logits.values())  # output_source_flow[0] + output_source_rgb[0]
        loss_verb = self.criterion(out_verb_fused, label['verb'])
        self.loss_verb.update(torch.mean(classification_weight * loss_verb) / (self.total_batch / self.batch_size),
                              self.batch_size)  # update with the normalized value

    def compute_egomo_loss(self, features):
        # Calculate egomo loss

        for m in self.modalities:
            if m == "Flow" and self.args.egomo > 0.0:
                feat_flow = features[m].detach()
            else:
                feat_egomo = features[m]
        loss_egomo = self.criterion_egomo(feat_egomo, feat_flow)
        self.loss_egomo.update((loss_egomo * self.args.egomo) / (self.total_batch / self.batch_size),
                              self.batch_size)  # update with the normalized value

    def compute_egomo_w_loss(self, features, logits, label):
        # Calculate egomo loss
        weight = 1
        for m in self.modalities:
            if m == "Flow" and self.args.egomo_w > 0.0:
                feat_flow = features[m].detach()
            else:
                feat_egomo = features[m]
                _, pred = logits[m].topk(1, 1, True, True)
                weight = 1 - ((pred == label["verb"].unsqueeze(1)).int())  # wrong prediction = 1
        loss_egomo = (self.criterion_egomo(feat_egomo, feat_flow) * weight).mean()

        self.loss_egomo_w.update((loss_egomo * self.args.egomo_w) / (self.total_batch / self.batch_size),
                              self.batch_size)  # update with the normalized value

    def compute_egomo_cossim_loss(self, features):
        # Calculate egomo cos sim loss

        for m in self.modalities:
            if m == "Flow" and self.args.egomo_cossim > 0.0:
                feat_flow = features[m].detach()
            else:
                feat_egomo = features[m]
        loss_egomo_cossim = 1 - self.criterion_egomo_cossim(feat_egomo, feat_flow).mean()
        self.loss_egomo_cossim.update((loss_egomo_cossim * self.args.egomo_cossim) / (self.total_batch / self.batch_size),
                              self.batch_size)  # update with the normalized value

    def compute_egomo_feat_patch_loss(self, features):
        # Calculate egomo loss

        for m in self.modalities:
            if m == "Flow" and self.args.egomo_feat_patch > 0.0:
                feat_flow = features[m].detach()
            else:
                feat_egomo = features[m]
        # feat_egomo  = (batch size, 1024, 2, 7, 7)

        alfa = random.randint(0,3)
        beta = random.randint(4,6)
        alfa1 = random.randint(0, 3)
        beta1 = random.randint(4, 6)

        feat_1 = feat_egomo[:, :, :, alfa:beta, alfa1:beta1].mean(2).mean(2).mean(2)
        feat_1_flow = feat_flow[:, :, :, alfa:beta, alfa1:beta1].mean(2).mean(2).mean(2).detach()
        loss_egomo_feat_patch = self.criterion_egomo_feat_patch(feat_1, feat_1_flow)



        self.loss_egomo_feat_patch.update((loss_egomo_feat_patch * self.args.egomo_feat_patch) / (self.total_batch / self.batch_size),
                              self.batch_size)  # update with the normalized value

    def compute_RNA_loss(self, features, weight_rna, args):

        '''MULTI RNA'''
        if args.rna:
            # import pdb; pdb.set_trace()
            radius = 1
            rgb_norm = (features['RGB']).norm(p=2, dim=1).mean()
            flow_norm = (features['Flow']).norm(p=2, dim=1).mean()
            print('RGB Norm: ', rgb_norm)
            print('Flow Norm: ', flow_norm)
            feat_frac = flow_norm / rgb_norm
            loss = get_L2norm_loss_self_driven(feat_frac, args.radius)  # RNA normale
            self.loss_rna.update(weight_rna * loss / (self.total_batch / self.batch_size),
                                 self.batch_size)
            del loss

    def compute_accuracy(self, output, label):
        # Apply softmax, to calculate accuracy
        # output = {m: nn.Softmax(dim=1)(out) for m, out in output.items()}
        # Fuse outputs from different modalities
        out_verb_fused = reduce(lambda x, y: x + y,
                                output.values())
        # Compute accuracy
        verb_prec1, verb_prec5, class_correct, class_total = utils.accuracy(out_verb_fused, label['verb'],
                                                                            topk=(1, 5))

        self.verb_top1.update(verb_prec1, self.batch_size, class_correct, class_total)
        self.verb_top5.update(verb_prec5, self.batch_size, class_correct, class_total)

        prec1, prec5 = verb_prec1, verb_prec5
        self.top1.update(prec1, self.batch_size)
        self.top5.update(prec5, self.batch_size)

    def log_stats(self, iter):
        if iter % LOG_FREQUENCY == 0:

            self.summaryWriter.add_scalar("train_loss_classification_verb_avg",
                                          self.loss_verb.avg * (self.total_batch / self.batch_size), iter + 1)
            self.summaryWriter.add_scalar("train_verb_top1_avg", self.verb_top1.avg, iter + 1)
            self.summaryWriter.add_scalar("train_loss_classification_verb_val",
                                          self.loss_verb.val * (self.total_batch / self.batch_size), iter + 1)
            self.summaryWriter.add_scalar("train_verb_top1_val", self.verb_top1.val, iter + 1)
            if self.args.egomo > 0.0:
                self.summaryWriter.add_scalar("loss_egomo_val", self.loss_egomo.val, iter + 1)
            if self.args.egomo > 0.0:
                self.summaryWriter.add_scalar("loss_egomo_avg", self.loss_egomo.avg, iter + 1)
                print("Loss-egomo ", self.loss_egomo.val * (self.total_batch / self.batch_size))

            if self.args.egomo_w > 0.0:
                self.summaryWriter.add_scalar("loss_egomo_w_val", self.loss_egomo_w.val, iter + 1)
            if self.args.egomo_w > 0.0:
                self.summaryWriter.add_scalar("loss_egomo_w_avg", self.loss_egomo_w.avg, iter + 1)
                print("Loss-egomo_w ", self.loss_egomo_w.val * (self.total_batch / self.batch_size))

            if self.args.egomo_cossim > 0.0:
                self.summaryWriter.add_scalar("loss_egomo_cossim_val", self.loss_egomo_cossim.val, iter + 1)
            if self.args.egomo_cossim > 0.0:
                self.summaryWriter.add_scalar("loss_egomo_cossim_avg", self.loss_egomo_cossim.avg, iter + 1)
                print("Loss-egomo_cossim ", self.loss_egomo_cossim.val * (self.total_batch / self.batch_size))

            if self.args.egomo_feat_patch > 0.0:
                self.summaryWriter.add_scalar("loss_egomo_feat_patch_val", self.loss_egomo_feat_patch.val, iter + 1)
            if self.args.egomo_feat_patch > 0.0:
                self.summaryWriter.add_scalar("loss_egomo_feat_patch_avg", self.loss_egomo_feat_patch.avg, iter + 1)
                print("Loss-egomo_feat_patch ", self.loss_egomo_feat_patch.val * (self.total_batch / self.batch_size))

            # log learning rate
            for m in self.modalities:
                self.summaryWriter.add_scalar("lr_" + m + '_' + self.name, self.optimizer[m].param_groups[0]["lr"],
                                              iter + 1)


    def log_gradients(self, iter):
        for m in self.modalities:
            for name, param in self.models[m].named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.summaryWriter.add_scalar("gradients_" + m + "/" + name, param.grad.norm(2).item(), iter)

    def reduce_learning_rate(self, update_audio_lr=False):
        if not update_audio_lr:
            for m in self.modalities:
                print('[{}] Reducing LR . . . {} --> '.format(m, self.optimizer[m].param_groups[-1]["lr"]), end='')
                self.optimizer[m].param_groups[-1]["lr"] = 0.001
                print(self.optimizer[m].param_groups[-1]["lr"])
        else:
            m = 'Spec'
            print('[{}] Reducing LR . . . {} --> '.format(m, self.optimizer[m].param_groups[-1]["lr"]), end='')
            self.optimizer['Spec'].param_groups[-1]["lr"] *= 0.1
            print(self.optimizer[m].param_groups[-1]["lr"])


    def reset_loss(self, rna=False):
        self.loss_verb.reset()
        if rna:
            if self.args.rna:
                self.loss_rna.reset()
        if self.args.egomo_cossim > 0.0:
            self.loss_egomo_cossim.reset()
        if self.args.egomo > 0.0:
            self.loss_egomo.reset()
        if self.args.egomo_feat_patch > 0.0:
            self.loss_egomo_feat_patch.reset()
        if self.args.egomo_w > 0.0:
            self.loss_egomo_w.reset()

    def reset_accMean(self):
        self.verb_top1.reset()
        self.verb_top5.reset()

    def backward(self, retain_graph):
        if self.args.egomo > 0.0:
            loss = self.loss_verb.val + (self.loss_egomo.val)
            loss.backward(retain_graph=retain_graph)
        elif self.args.egomo_w > 0.0:
            loss = self.loss_verb.val + (self.loss_egomo_w.val)
            loss.backward(retain_graph=retain_graph)
        elif self.args.egomo_cossim > 0.0:
            loss = self.loss_verb.val + (self.loss_egomo_cossim.val)
            loss.backward(retain_graph=retain_graph)
            self.loss_egomo_cossim.reset()
        elif self.args.egomo_feat_patch > 0.0:
            loss = self.loss_verb.val + (self.loss_egomo_feat_patch.val)
            loss.backward(retain_graph=retain_graph)
            self.loss_egomo_feat_patch.reset()
        else:
            self.loss_verb.val.backward(retain_graph=retain_graph)
    def backward_RNA(self, args, retain_graph):
        self.loss_rna.val.backward(retain_graph=retain_graph)
