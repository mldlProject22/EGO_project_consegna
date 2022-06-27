from datetime import datetime
from pathlib import Path

import torch.nn.parallel
import torch.optim
from utils.loaders import VideoDataset
from utils.args import parser
from tensorboardX import SummaryWriter
import warnings
from utils.utils import get_model_spec, sync_train_test_model
from models.task import ActionRecognition
from torch.nn.utils import clip_grad_norm_
import time
import utils
import numpy as np
import pandas as pd
import os

# suppress all warnings
warnings.filterwarnings("ignore")

training_iterations = 0
np.random.seed(13696641)
torch.manual_seed(13696641)

# parser is from opts!
args = parser.parse_args()
experiment_dir = os.path.join(args.name, datetime.now().strftime('%b%d_%H-%M-%S'))
log_dir = os.path.join('E2GOMOTION', experiment_dir)
summaryWriter = SummaryWriter(logdir=log_dir)
if args.gpus is not None:
    print('Using only these GPUs: {}'.format(args.gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

ACTION_CLASSIFIER = 'action-classifier'  # it is the whole network: backbone + final classifier

KD_from_flow = (args.egomo > 0) or \
               (args.egomo_cossim > 0) or \
               (args.egomo_feat_patch > 0) or \
               (args.egomo_w > 0) #add here others KD version
if KD_from_flow:
    print("KD FROM FLOW IS ON")


def main():
    global args, experiment_dir, training_iterations
    modalities = args.modality
    utils.utils.take_path(args)
    print('\n---------- ARGUMENTS ----------\n')
    # print all the arguments on file and on the terminal
    with open(os.path.join(log_dir, 'parameters.txt'), 'w') as myfile:
        for arg in sorted(vars(args)):
            myfile.write('{:>20} = {}\n'.format(arg, getattr(args, arg)))
            print('{:>20} = {}'.format(arg, getattr(args, arg)))

    ''' CLASSES USED FOR THIS SETTING
        Verbs:
        0 - take (get)
        1 - put-down (put/place)
        2 - open
        3 - close
        4 - wash (clean)
        5 - cut
        6 - stir (mix)
        7 - pour
        Domains:
        D1 - P08
        D2 - P01
        D3 - P22
    '''

    # recover valid paths, domains, classes
    # this will output the domain conversion (D1 -> 8, et cetera) and the label list
    num_classes, valid_domains, valid_labels, source_domains, target_domains, target_test = utils.utils.get_domains_and_labels(args)

    # define number of iterations I'll do with the actual batch: we do not reason with epochs but with iterations
    # i.e. number of batches passed
    # notice, here it is multiplied by tot_batch/batch_size since gradient accumulation is used
    training_iterations = args.num_iter * (args.total_batch // args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ''' ---------- Init models for all the different tasks ----------'''
    print('\n---------- INIT MODELS ----------\n')
    # --- Action classifier
    print('Initializing Action Classifier . . .\nLR = {}'.format(args.lr))
    model = {m: args.model[modalities.index(m)] for m in modalities}
    dense_sampling_train = {m: args.dense_sampling_train[modalities.index(m)] for m in modalities}
    dense_sampling_test = {m: args.dense_sampling_test[modalities.index(m)] for m in modalities}
    # arch is a dictionary where the key is the modality and the value is the network associated to that modality
    arch = {m: args.base_arch[modalities.index(m)] for m in modalities}

    model_spec = {m: get_model_spec(model[m], args, m) for m in modalities}
    num_frames_per_clip_train = {m: model_spec[m]["num_frames_per_clip_train"] for m in modalities}
    num_frames_per_clip_test = {m: model_spec[m]["num_frames_per_clip_test"] for m in modalities}

    action_classifier = {m: model_spec[m]["net"](num_classes, model_spec[m]["num_frames_per_clip_train"], m,
                                                 base_model=arch[m],
                                                 args=args,
                                                 **model_spec[m]["kwargs"])
                         for m in modalities}
    # create optimizer on these parameters. Return a Task object (inside it will have the dict name: Net)
    action_classifier = utils.utils.init_model(action_classifier, ACTION_CLASSIFIER, args, summaryWriter)
    modalitis_without_flow = [m for m in modalities if m != "Flow"]
    modalities_test = modalitis_without_flow if KD_from_flow else modalities

    action_classifier_test = {
        m: torch.nn.DataParallel(model_spec[m]["net"](num_classes, model_spec[m]["segments_test"], m,
                                                      base_model=arch[m],
                                                      args=args,
                                                      **model_spec[m]["kwargs"])).to(device)
        for m in modalities_test}

    # ActionRecognition is a subclass of Task. Task has inside .step, .zerograd ecc ecc. So in the train loop I will
    # cycle on tasks and e.g. .step will be done on all the blocks.
    action_classifier = ActionRecognition(action_classifier, source_domains, target_domains)
    # tasks is a dictionary of Task of type ActionRecognition
    tasks = {action_classifier.name: action_classifier}
    print('DONE\n')

    # Construct data loaders
    print('\n------ CREATING DATA LOADERS ------\n')

    # The following returns train/val dictionaries containing the transformations (augmentation+normalization)
    # together with dictionaries containing image name templates for each modality
    image_tmpl, train_transform, val_transform = utils.utils.data_preprocessing(action_classifier.models, modalities,
                                                                    args.flow_prefix, args)

    train_loader_source = torch.utils.data.DataLoader(
        VideoDataset(pd.read_pickle(args.train_list),
                     args.modality,
                     image_tmpl,
                     num_frames_per_clip=num_frames_per_clip_train,
                     dense_sampling=dense_sampling_train,
                     fixed_offset=False,
                     visual_path=args.visual_path,
                     flow_path=args.flow_pwc_path if args.pwc else args.flow_path,
                     event_path=args.event_path,
                     num_clips=1,
                     mode='train',
                     transform=train_transform,
                     args=args),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # UDA or DG
    if args.DG or args.UDA:
        pkl = args.train_list2 if args.DG else args.train_list_target

        train_loader_2 = torch.utils.data.DataLoader(
            VideoDataset(pd.read_pickle(pkl),
                         args.modality,
                         image_tmpl,
                         num_frames_per_clip=num_frames_per_clip_train,
                         dense_sampling=dense_sampling_train,
                         fixed_offset=False,
                         visual_path=args.visual_path,
                         flow_path=args.flow_pwc_path if args.pwc else args.flow_path,
                         event_path=args.event_path,
                         num_clips=1,
                         mode='train',
                         transform=train_transform,
                         args=args),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        train_loader_2 = None

    val_loader = torch.utils.data.DataLoader(
        VideoDataset(pd.read_pickle(args.val_list),
                     args.modality,
                     image_tmpl,
                     num_frames_per_clip=num_frames_per_clip_test,
                     dense_sampling=dense_sampling_test,
                     fixed_offset=True,
                     visual_path=args.visual_path,
                     flow_path=args.flow_pwc_path if args.pwc else args.flow_path,
                     event_path=args.event_path,
                     num_clips=args.num_clips_test,
                     mode='val',
                     transform=val_transform,
                     args=args),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    print("TRAIN SOURCE SAMPLES = {}\n"
          "VAL SAMPLES = {}".format(len(train_loader_source) * args.batch_size,
                                    len(val_loader) * (args.batch_size)))

    # cycle on our models and initialize them
    for task in tasks.values():
        task.load_on_GPU(device)
        iteration, task.best_iter, task.best_iter_score, task.acc_mean, task.loss_mean = 0, 0, 0, 0, 0

    # Resume weights models from a checkpoint
    if args.resume_from is not None:
        print('\n---------- RESTORE ----------\n')
        print('Restoring from \'{}\'\n'.format(args.resume_from))
        for task in tasks.values():
            iteration = task.resume_model(args.resume_from)
        print('\nRestarting from iteration [{}/{}]\nBest val accuracy = {:.2f}%'.format(
            iteration,
            training_iterations,
            action_classifier.best_iter_score))

    # Resume weights models from a checkpoint
    if args.resume_from_iteration is not None:
        print('\n---------- RESTORE ----------\n')
        print('Restoring from \'{}\'\n'.format(args.resume_from_iteration))
        for task in tasks.values():
            iteration = task.resume_model_iteration(args.resume_from_iteration, args.iteration)
        print('\nRestarting from iteration [{}/{}]\nBest val accuracy = {:.2f}%'.format(
            iteration,
            training_iterations,
            action_classifier.best_iter_score))

    ##############
    #    egomo    #
    ##############
    if KD_from_flow and args.model[0] == 'TSM':
        # IF egomo --> load weight of Flow-net pretrain and freeze them
        path_model_flow = args.egomo_path_flow
        shift_model = args.shift.split("-")[0]  # D1 or D2 or D3
        path_model_flow = path_model_flow + shift_model + "/" +"5000_action-classifier_Flow_9.pth"
        flow_model = path_model_flow
        #saved_models = [x for x in reversed(sorted(Path(path_model_flow).iterdir(), key=os.path.getmtime))]
        #flow_model = list(filter(lambda x: "Flow" == x.name.split('.')[0].split('_')[-2]  # modality check
        #                                   and "5000" == x.name.split('.')[0].split('_')[-4],
        #                         saved_models))[0]

        print('\n----------***   RESTORE FLOW MODEL for TSM egomo  ***----------\n')
        print('Restoring Flow-Model from \'{}\'\n'.format(flow_model))
        for task in tasks.values():
            task.resume_flow_model(flow_model)

        if (args.flow_classifier_init) and ("Flow" in modalities) and (args.resume_from is None): #training and not resume stop training
            for task in tasks.values():
                task.resume_from_flow_model(flow_model)



    if KD_from_flow and args.model[0] == 'i3d':
        # IF egomo --> load weight of Flow-net pretrain and freeze them
        path_model_flow = args.egomo_path_flow
        shift_model = args.shift.split("-")[0]  # D1 or D2 or D3
        path_model_flow = path_model_flow + shift_model
        saved_models = [x for x in reversed(sorted(Path(path_model_flow).iterdir(), key=os.path.getmtime))]
        flow_model = list(filter(lambda x: "Flow" == x.name.split('.')[0].split('_')[-2]  # modality check
                                           and "5000" == x.name.split('.')[0].split('_')[-4],
                                 saved_models))[0]
        print('\n----------***   RESTORE FLOW MODEL (I3D) for egomo  ***----------\n')
        print('Restoring Flow-Model from \'{}\'\n'.format(flow_model))
        for task in tasks.values():
            task.resume_flow_model(flow_model)

    print('\n---------- TRAINING START ----------\n')
    train(tasks, train_loader_source, val_loader,
          device, args.num_clips_test,
          iteration, test_models=action_classifier_test, second_loader=train_loader_2 if args.UDA or args.DG else None)


def train(tasks, train_loader, val_loader,
          device, num_clips_test, iteration, test_models=None, second_loader=None):
    global training_iterations

    batch_size = args.batch_size
    modalities = args.modality

    # I assign the objects of the dict tasks to specific variables
    action_classifier = tasks[ACTION_CLASSIFIER]
    data_loader_source = iter(train_loader)
    # UDA or DG
    if second_loader is not None:
        data_loader = iter(second_loader)

    # switch to train mode and zero all gradients before starting the training process
    for task in tasks.values():
        task.train(True)
        task.zero_grad()

    Cont = 0  # to save 9 models

    # the batch size should be 128 (it is the default value of total_batch) but batch accumulation is done.
    # real_iter is the number of iterations if the batch size was really 128 (which is the bs we want to simulate)
    for i in range(iteration, training_iterations):
        # iteration w.r.t. the paper (w.r.t the bs to simulate).... i is the iteration with the actual bs( < tot_bs)
        real_iter = i / (args.total_batch // args.batch_size)
        to_do_step = list(tasks.keys())

        if real_iter == args.lr_steps:
            print("LR STEP for Visual")
            for task in to_do_step:
                tasks[task].reduce_learning_rate()

        start_t = datetime.now()
        if args.verbose:
            print("[%s] Retrieving %s batch..." % (start_t.strftime("%H:%M:%S"), 'batch'))

        # the dataloaders have different dimension so when one of them finishes, it must be restarted
        try:
            source_data, source_label = next(data_loader_source)

        except StopIteration:
            print('except source A')
            data_loader_source = iter(train_loader)
            source_data, source_label = next(data_loader_source)
            # reset accuracy
            for task in to_do_step:
                tasks[task].reset_accMean()

        if args.UDA or args.DG: # second loader
            try:
                data, label = next(data_loader)

            except StopIteration:
                print('except data B')
                data_loader = iter(second_loader)
                data, source = next(data_loader)
                # reset accuracy
                for task in to_do_step:
                    tasks[task].reset_accMean()

        RNA = False
        if args.rna:
            RNA = True

        end_t = datetime.now()

        if args.verbose:
            print("[%s] Batch %s retrieved! Elapsed time = %d m %d s" % (end_t.strftime("%H:%M:%S"),
                                                                         'batch',
                                                                         (end_t - start_t).total_seconds() // 60,
                                                                         (end_t - start_t).total_seconds() % 60))

        ''' Action recognition'''
        source_label = {k: v.to(device) for k, v in source_label.items()}

        for m in modalities:
            source_data[m] = source_data[m].to(device)



        weight = 1

        ###################
        #     Source      #
        ###################

        # we pass the batch, the labels and it calculates accuracy, loss, backward ecc ecc
        # retain_graph is needed to avoid losing dependencies on Variables after first backward
        _, feat_source, feat_source_spatial, _ = action_classifier.execute_task(input=source_data, label=source_label,
                                                              retain_graph=RNA,
                                                              is_target=False,
                                                              weight_class=weight)

        if RNA:
            action_classifier.compute_RNA_loss(feat_source, weight_rna=args.weight_rna, args=args)
            action_classifier.backward_RNA(
                retain_graph= False,
                args=args)

        del feat_source, feat_source_spatial, source_data

        ####################
        #      Data 2      #
        #     UDA or DG    #
        ####################

        if args.UDA or args.DG:
            is_target_data = True if args.UDA else False
            for m in modalities:
                data[m] = data[m].to(device)
            if args.DG:
                label = {k: v.to(device) for k, v in label.items()}

            _, feat, feat_spatial, _ = action_classifier.execute_task(input=data, label=label,
                                                                                retain_graph=RNA,
                                                                                is_target=is_target_data,
                                                                                weight_class=weight)

            if RNA:
                action_classifier.compute_RNA_loss(feat, weight_rna=args.weight_rna, args=args)
                action_classifier.backward_RNA(
                    retain_graph=False,
                    args=args)
            del feat, feat_spatial, data

        # update weights and zero gradients
        if (i + 1) % (args.total_batch / batch_size) == 0:

            print(
                "[%d/%d]\tlast Verb loss: %.4f\tMean verb loss: %.4f\tAcc@1: %.2f%%\tAccMean@1: %.2f%%" %
                ((i + 1) / (args.total_batch / batch_size),
                 training_iterations / (args.total_batch / batch_size),
                 action_classifier.loss_verb.val * (args.total_batch / batch_size),
                 action_classifier.loss_verb.avg * (args.total_batch / batch_size),
                 action_classifier.verb_top1.val,
                 action_classifier.verb_top1.avg))
            with open(os.path.join(log_dir, 'loss_accumul_log.txt'), 'a+') as f:
                f.write("[%d/%d]\tloss: %.4f\tAcc@1: %.2f%%\n" %
                        (i + 1, training_iterations,
                         action_classifier.loss_verb.val * (args.total_batch / batch_size),
                         action_classifier.verb_top1.val))

            for task in tasks:
                for m in tasks[task].modalities:
                    for name, param in tasks[task].models[m].named_parameters():
                        if param.requires_grad and param.grad is not None:
                            if param.grad.norm(2).item() > 25:
                                print(name)
            if args.clip_gradient is not None:
                for task in tasks:
                    for m in tasks[task].modalities:
                        total_norm = clip_grad_norm_(tasks[task].models[m].parameters(), args.clip_gradient)
                        if total_norm > args.clip_gradient:
                            print("clipping gradient: {} with coef {}".format(total_norm,
                                                                              args.clip_gradient / total_norm))
            for task in to_do_step:
                print(to_do_step)
                tasks[task].step()
                tasks[task].log_stats(i)
                # some losses are reset not to full the memory
                tasks[task].reset_loss(RNA)

            action_classifier.log_gradients(i)
            for task in to_do_step:
                tasks[task].zero_grad()

        if ((i + 1) % (1000 * (args.total_batch / batch_size))) == 0:
            for name, task in tasks.items():
                print('Saving {} model at iter [{}/{}]'.format(name, i + 1, training_iterations))
                task.save_model(i, task.best_iter, task.best_iter_score,
                                task.verb_top1 if name == ACTION_CLASSIFIER else None,
                                task.loss_verb if name == ACTION_CLASSIFIER else task.loss,
                                experiment_dir, Cont if name == ACTION_CLASSIFIER else 0, task.optimizer,
                                str(int((i + 1) / (args.total_batch / batch_size))))

        # Save model before updating learning step
        elif real_iter == (args.lr_steps - 1):
            for name, task in tasks.items():
                print('Saving {} model at iter [{}/{}]'.format(name, i + 1, training_iterations))
                task.save_model(i, task.best_iter, task.best_iter_score,
                                task.verb_top1 if name == ACTION_CLASSIFIER else None,
                                task.loss_verb if name == ACTION_CLASSIFIER else task.loss,
                                experiment_dir, Cont if name == ACTION_CLASSIFIER else 0, task.optimizer,
                                str(args.lr_steps))

        flag_val = False
        if real_iter <= 50 or real_iter > 5000 or ((i + 1) % (1000 * (args.total_batch / batch_size))) == 0:
            flag_val = True

        # every eval_freq "real iteration" (iterations on total_batch) the validation is done
        # we save every 9 models (dima policy: to validate, it takes the last 9 models, it tests them all and then
        # it computes the average. This is done to avoid peaks in the performances)
        if ((i + 1) % (args.eval_freq * (args.total_batch / batch_size))) == 0:
            Cont = Cont + 1 if Cont < 9 else 1
            if flag_val:
                val_metrics = validate(val_loader, action_classifier.models, test_models, device, num_clips_test)
                summaryWriter.add_scalar("val_loss_classification_verb",
                                         val_metrics['loss'], real_iter + 1)
                summaryWriter.add_scalar("val_verb_top1", val_metrics['top1'], real_iter + 1)
                with open(os.path.join(log_dir, 'val_precision.txt'), 'a+') as f:
                    f.write("[%d/%d]\tAcc@top1: %.2f%%\n" %
                            ((i + 1) / (args.total_batch / batch_size),
                             training_iterations / (args.total_batch / batch_size),
                             val_metrics['verb_top1']))

                if val_metrics['verb_top1'] <= action_classifier.best_iter_score:
                    print("Accuracy not updated...\nCurrent best VERB accuracy {:.2f}%"
                          .format(action_classifier.best_iter_score))
                else:
                    print("Best accuracy so far!")
                    action_classifier.best_iter = i
                    action_classifier.best_iter_score = val_metrics['verb_top1']

            # Save checkpoints for all models
            for name, task in tasks.items():
                print('Saving {} model at iter [{}/{}]'.format(name, i + 1, training_iterations))
                task.save_model(i, task.best_iter, task.best_iter_score,
                                task.verb_top1 if name == ACTION_CLASSIFIER else None,
                                task.loss_verb if name == ACTION_CLASSIFIER else task.loss,
                                experiment_dir, Cont if name == ACTION_CLASSIFIER else 0, task.optimizer, '')
                # 'Cont if name == ACTION_CLASSIFIER_NAME else 0' is for overwriting models for which
                # we do not need multiple saved models

            for task in tasks.values():
                task.train(True)


# during the validation we take 5 clips of the video and we average on them (so we have a dimension more)
def validate(val_loader, model_list_train, model_list_test, device, num_clips_test):
    if model_list_train is not None:
        sync_train_test_model(model_list_train, model_list_test)
    channels = {"RGB": 3, "Flow": 2, "Event": args.channels_events}
    model_list = model_list_test
    # when validating during train, we don't have a list of models, so we create it
    if not isinstance(model_list, list):
        model_list = [model_list]
    modalities = model_list[0].keys()

    # just for action-recognition
    criterion = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                          reduce=None, reduction='none')
    classification_weight = 1.

    with torch.no_grad():

        top1 = utils.utils.AverageMeter()
        top5 = utils.utils.AverageMeter()
        verb_top1 = utils.utils.AverageMeter()
        verb_top5 = utils.utils.AverageMeter()
        loss_verb = utils.utils.AverageMeter()
        batch_time = utils.utils.AverageMeter()

        # switch to evaluate mode
        for model in model_list:
            for m in modalities:
                model[m].train(False)

        end = time.time()

        N_CLASSES = 8

        accuracies = {'top1': [], 'top5': [], 'verb_top1': [], 'verb_top5': [], 'loss': []}
        class_accuracies = []
        # Iterate over the models
        for j, model in enumerate(model_list):
            print('Testing model %d . . .' % j)
            # get batch
            top1.reset()
            top5.reset()
            verb_top1.reset()
            verb_top5.reset()
            loss_verb.reset()
            for i_val, (input, label) in enumerate(val_loader):
                for m in modalities:
                    input[m] = input[m].to(device)
                    B, C, H, W = input[m].shape
                    input[m] = input[m].reshape(B, num_clips_test, -1, H, W)
                    input[m] = input[m].permute(1, 0, 2, 3, 4)
                label = {k: v.to(device) for k, v in label.items()}

                outputs = torch.zeros((len(modalities), len(input[list(modalities)[0]]), B, N_CLASSES)).to(device)
                for i_m, m in enumerate(modalities):
                    for i_c, clip in enumerate(input[m]):
                        if model[m].module.name == "TSN":
                            clip = clip.contiguous().view(-1, channels[m], clip.size(2), clip.size(3))
                        if model[m].module.name == "TSM":
                            clip = clip.contiguous().view(label['verb'].numel(),
                                                          get_model_spec("TSM", args, m)["segments_test"],
                                                          channels[m], clip.size(2), clip.size(3))
                        out, _, _, _ = model[m](input=clip, is_target=True)
                        if model[m].module.name == "TSM":
                            out = out.reshape(label['verb'].numel(), N_CLASSES)
                        elif model[m].module.name == "TSN":
                            # fai la media dei segmenti
                            out = out.reshape((label['verb'].numel(), -1, N_CLASSES)).mean(axis=1) \
                                .reshape((label['verb'].numel(), N_CLASSES))
                        outputs[i_m, i_c, :, :] = out

                outputs = torch.sum(outputs, dim=0)  # fuse the modalities by summing their outputs
                outputs = torch.mean(outputs, dim=0)  # fai la media dei logits dei clips
                loss_calc = criterion(outputs, label['verb'])
                verb_prec1, verb_prec5, class_correct, class_total = utils.utils.accuracy(outputs, label['verb'],
                                                                              topk=(1, 5))

                # added to compute loss together with accuracy in validation
                loss_verb.update(torch.mean(classification_weight * loss_calc), B)  # update with the normalized value

                # Update values
                verb_top1.update(verb_prec1, B, class_correct, class_total)
                verb_top5.update(verb_prec5, B, class_correct, class_total)
                top1.update(verb_prec1, B, class_correct, class_total)
                top5.update(verb_prec5, B, class_correct, class_total)
                if (i_val + 1) % (len(val_loader) // 5) == 0:
                    print("MODEL {} [{}/{}] top1 avg = {:.3f}% top5 = {:.3f}% loss = {:f}".format(j, i_val + 1,
                                                                                                  len(val_loader),
                                                                                                  top1.avg,
                                                                                                  top5.avg,
                                                                                                  loss_verb.avg))

            accuracies['top1'].append(top1.avg)
            accuracies['top5'].append(top5.avg)
            accuracies['verb_top1'].append(verb_top1.avg)
            accuracies['verb_top5'].append(verb_top5.avg)
            accuracies['loss'].append(loss_verb.avg)
            class_accuracies.append([(x / y) * 100 for x, y in zip(top1.correct, top1.total)])
            print('----- END model %d!\ttop1 = %.2f%%\ttop5 = %.2f%%\nCLASS accuracy:' % (j, top1.avg, top5.avg))
            for i_class, class_acc in enumerate(class_accuracies[j]):
                print('Class %d = [%d/%d] = %.2f%%' % (i_class,
                                                       int(top1.correct[i_class]),
                                                       int(top1.total[i_class]),
                                                       class_acc))

            batch_time.update(time.time() - end)
            end = time.time()

        for i, c in enumerate(np.array(class_accuracies).mean(axis=0)):
            print('Accuracy of {} : {}%'.format(i, c))
        test_results = {'verb_top1': sum(accuracies['verb_top1']) / float(len(accuracies['verb_top1'])),
                        'verb_top5': sum(accuracies['verb_top5']) / float(len(accuracies['verb_top5'])),
                        'top1': sum(accuracies['top1']) / float(len(accuracies['top1'])),
                        'top5': sum(accuracies['top5']) / float(len(accuracies['top5'])),
                        'class_accuracies': np.array(class_accuracies).mean(axis=0),
                        'loss': sum(accuracies['loss']) / float(len(accuracies['loss']))}
        message = ("- - - - - - - - Testing Results:\n"
                   "Verb Prec@1\t{:.3f}%\nVerb Prec@5\t{:.3f}%\n"
                   "Prec@1\t{:.3f}%\nPrec@5\t{:.3f}%\n"
                   "Loss\t{:f}\n"
                   "Class accuracies\n{}"
                   ).format(test_results['verb_top1'],
                            test_results['verb_top5'],
                            test_results['top1'],
                            test_results['top5'],
                            test_results['loss'],
                            test_results['class_accuracies']
                            )
        print(message)

        return test_results


if __name__ == '__main__':
    main()
