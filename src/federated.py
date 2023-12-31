import utils
import models
import math
import copy
import numpy as np
from agent import Agent
from agent_sparse import Agent as Agent_s
from options import args_parser
from aggregation import Aggregation
# from torch.utils.tensorboard import SummaryWriter
import torch
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm
import poison_cifar as poison
from torch.utils.data import DataLoader, RandomSampler
from optimize_mask_cifar import train_mask
from prune_neuron_cifar import pruning
import time
import logging
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import random_split, DataLoader, Dataset
import os
from collections import OrderedDict
from anp_batchnorm import NoisyBatchNorm2d




# SAVE_MODEL_NAME = 'combined_train.pt'
SAVE_MODEL_NAME = 'AckRatio4_40_MethodNone_datacifar10_alpha1_Rnd200_Epoch2_inject0.5_dense0.25_Aggavg_se_threshold0.0001_noniidTrue_maskthreshold20_attackbadnet.pt'
import torch.nn as nn

def replace_batchnorm_with_noisy(model):
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Replace with NoisyBatchNorm2d
            setattr(model, name,  NoisyBatchNorm2d(module.num_features, 
                                                  eps=module.eps, 
                                                  momentum=module.momentum, 
                                                  affine=module.affine, 
                                                  track_running_stats=module.track_running_stats))
        else:
            # Recursively apply to child modules
            replace_batchnorm_with_noisy(module)



def get_train_loader(args):
    print('==> Preparing train data..')
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.RandomRotation(3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    if (args.dataset == 'CIFAR10'):
        trainset = datasets.CIFAR10(root='data/CIFAR10', train=True, download=True)
    else:
        raise Exception('Invalid dataset')

    train_data = DatasetCL(args, full_dataset=trainset, transform=tf_train)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    return train_loader

def train_step_unlearning(args, model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    device = 'cuda:0'
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        (-loss).backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [param for name, param in unlearned_model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)

def train_step_recovering(args, unlearned_model, criterion, mask_opt, data_loader):
    unlearned_model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        mask_opt.zero_grad()
        output = unlearned_model(images)
        loss = criterion(output, labels)
        loss = args.alpha * loss

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(unlearned_model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)

def global_train(args,global_model, criterion, round=None, neurotoxin_mask=None):
    """ Do a local training over the received global model, return the update """
    global_model.train()
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.client_lr, weight_decay=args.wd, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    running_loss = 0.0
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    for round_index in tqdm(range(round)):
        for _, (inputs, labels) in enumerate(args.combined_train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device=args.device, non_blocking=True), \
                                labels.to(device=args.device, non_blocking=True)
            outputs = global_model(inputs)
            minibatch_loss = criterion(outputs, labels)
            minibatch_loss.backward()
            optimizer.step()
            running_loss += minibatch_loss.item()
            # print(minibatch_loss)
        # At the end of the epoch, compute and print the average loss
        average_loss = running_loss / len(args.combined_train_loader)
        print(f'Epoch {round_index+1}, Average Loss: {average_loss:.4f}')
        scheduler.step()
        if round_index % 10 == 0:
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader,args, rnd, num_target)
                print(f'| Val_Acc:   {val_acc:.3f} |')
                acc_vec.append(val_acc)
                per_class_vec.append(val_per_class_acc)

                poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,poisoned_val_loader, args, rnd, num_target)
                asr_vec.append(asr)
                print(f'| Attack Success Ratio: {asr:.3f} |')
    global_model.eval()
    return global_model

class DatasetCL(Dataset):
    def __init__(self, args, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=args.ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset

    
if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True

    args = args_parser()
    args.server_lr = args.server_lr if args.aggr == 'sign' else 1.0
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.DEBUG)
    if not args.debug:
        logPath = "logs"
        if args.mask_init == "uniform":
            fileName = "uniformAckRatio{}_{}_Method{}_data{}_alpha{}_Epoch{}_inject{}_dense{}_Agg{}_same_mask{}_noniid{}_maskthreshold{}_attack{}_af{}.pt.pt".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, args.local_ep, args.poison_frac,
                args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,args.anneal_factor)
        elif args.dis_check_gradient == True:
            fileName = "NoGradientAckRatio{}_{}_Method{}_data{}_alpha{}_Epoch{}_inject{}_dense{}_Agg{}_same_mask{}_noniid{}_maskthreshold{}_attack{}_af{}.pt.pt".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, args.local_ep, args.poison_frac,
                args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,args.anneal_factor)
        else:
            fileName = "AckRatio{}_{}_Method{}_data{}_alpha{}_Epoch{}_inject{}_dense{}_Agg{}_same_mask{}_noniid{}_maskthreshold{}_attack{}_endpoison{}_af{}.pt".format(
                args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, args.local_ep, args.poison_frac,
                args.dense_ratio, args.aggr, args.same_mask, args.non_iid, args.theta, args.attack,
                args.cease_poison, args.anneal_factor)
        fileHandler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName))
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    # sys.stderr.write = rootlogging.error
    # sys.stdout.write = rootlogging.info
    # consoleHandler = logging.StreamHandler(sys.stdout)
    # consoleHandler.setFormatter(logFormatter)
    # rootlogging.addHandler(consoleHandler)
    logging.info(args)

    cum_poison_acc_mean = 0

    # load dataset and user groups (i.e., user to data mapping)
    train_dataset, val_dataset = utils.get_datasets(args.data)
    train_dataset_copy = copy.deepcopy(train_dataset)
    if args.data == "cifar100":
        num_target = 100
    elif args.data == "tinyimagenet":
        num_target = 200
    else:
        num_target = 10
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                            pin_memory=False)
    # fedemnist is handled differently as it doesn't come with pytorch
    if args.data != 'fedemnist':
        if args.non_iid:
            user_groups = utils.distribute_data_dirichlet(train_dataset, args)
        else:
            user_groups = utils.distribute_data(train_dataset, args, n_classes=num_target)
        # print(user_groups)
    idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
    # logging.info(idxs)
    if args.data != "tinyimagenet":
        # poison the validation dataset
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set.dataset, args, idxs, poison_all=True)
    else:
        poisoned_val_set = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args)

    poisoned_val_loader = DataLoader(poisoned_val_set, batch_size=args.bs, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=False)

    # poison the validation dataset
    # logging.info(idxs)
    if args.data != "tinyimagenet":
        idxs = (val_dataset.targets != args.target_class).nonzero().flatten().tolist()
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs)
        utils.poison_dataset(poisoned_val_set_only_x.dataset, args, idxs, poison_all=True, modify_label=False)
    else:
        poisoned_val_set_only_x = utils.DatasetSplit(copy.deepcopy(val_dataset), idxs, runtime_poison=True, args=args, modify_label =False)

    poisoned_val_only_x_loader = DataLoader(poisoned_val_set_only_x, batch_size=args.bs, shuffle=False,
                                            num_workers=args.num_workers,
                                            pin_memory=False)

    # initialize a model, and the agents
    global_model = models.get_model(args.data).to(args.device)
    if args.rounds == 0 or args.rounds == -1:
        global_model.load_state_dict(torch.load(f'./checkpoint/{SAVE_MODEL_NAME}')['model_state_dict'])

        # global_model_1 = copy.deepcopy(global_model)
        # global_model_2 = copy.deepcopy(global_model)
        # SAVE_MODEL_NAME = 'AckRatio4_40_MethodNone_datacifar10_alpha1_Rnd200_Epoch2_inject0.5_dense0.25_Aggavg_se_threshold0.0001_noniidTrue_maskthreshold20_attackbadnet.pt'
        # SAVE_MODEL_NAME = 'AckRatio0_40_MethodNone_datacifar10_alpha1_Rnd200_Epoch2_inject0.5_dense0.25_Aggavg_se_threshold0.0001_noniidTrue_maskthreshold20_attackbadnet_prox.pt'
        # global_model_1.load_state_dict(torch.load(f'/work/LAS/wzhang-lab/mingl/code/backdoor/no_lockdown/src/checkpoint/{SAVE_MODEL_NAME}')['model_state_dict'])
        # SAVE_MODEL_NAME = 'combined_train.pt'
        # SAVE_MODEL_NAME = 'combined_train_prox.pt'
        # global_model_2.load_state_dict(torch.load(f'/work/LAS/wzhang-lab/mingl/code/backdoor/no_lockdown/src/checkpoint/{SAVE_MODEL_NAME}')['model_state_dict'])
        # # Assuming 'global_model_1' and 'global_model_2' are your PyTorch models
        # parameter_vector_1 = parameters_to_vector(global_model_1.parameters()).detach()
        # parameter_vector_2 = parameters_to_vector(global_model_2.parameters()).detach()

        # # Convert to NumPy arrays for plotting
        # weights_numpy_1 = parameter_vector_1.cpu().numpy()
        # weights_numpy_2 = parameter_vector_2.cpu().numpy()

        # # Define bin edges with more bins near zero
        # bins = [-1, -0.5, -0.1] + [i * 0.01 for i in range(-10, 11)] + [0.1, 0.5, 1]

        # # Plot the distribution of weights for both vectors
        # plt.figure(figsize=(10, 6))
        # plt.hist(weights_numpy_1, bins=bins, color='blue', alpha=0.7, label='Model fl')
        # plt.hist(weights_numpy_2, bins=bins, color='red', alpha=0.7, label='Model central')
        # plt.title('Distribution of Weights for Two Models')
        # plt.xlabel('Weight Value')
        # plt.ylabel('Frequency')
        # plt.legend(loc='upper right')
        # plt.grid(True)

        # # Save the figure
        # plt.savefig('comparison_weights_distribution_clean.png', bbox_inches='tight')

        # If you want to display it as well, uncomment the next line
        # plt.show()

        # After saving, you can close the plot if it's not needed to be shown
        # plt.close()
        # exit()

    global_mask = {}
    neurotoxin_mask = {}
    updates_dict = {}
    n_model_params = len(parameters_to_vector([ global_model.state_dict()[name] for name in global_model.state_dict()]))
    params = {name: copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()}
    if args.method == "lockdown":
        sparsity = utils.calculate_sparsities(args, params, distribution=args.mask_init)
        mask = utils.init_masks(params, sparsity)
    agents, agent_data_sizes = [], {}
    for _id in range(0, args.num_agents):
        if args.data == 'fedemnist':
            if args.method == "lockdown":
                agent = Agent_s(_id, args, mask=mask)
            else:
                agent = Agent(_id, args)
        else:
            if args.method == "lockdown":
                if args.same_mask==0:
                    agent = Agent_s(_id, args, train_dataset, user_groups[_id], mask=utils.init_masks(params, sparsity))
                else:
                    agent = Agent_s(_id, args, train_dataset, user_groups[_id], mask=mask)
            else:
                agent = Agent(_id, args, train_dataset, user_groups[_id])
        agent_data_sizes[_id] = agent.n_data
        agents.append(agent)

        # aggregation server and the loss function

    aggregator = Aggregation(agent_data_sizes, n_model_params, poisoned_val_loader, args, None)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # mask = { name: torch.ones(param)   for name, param in global_model.named_parameters() }

    # training loop
    # for name in torch

    # mask [:int(0.01*n_model_params)] =1
    # mask = torch.ones( n_model_params)
    server_update = torch.zeros_like(parameters_to_vector(global_model.parameters()).detach())
    agent_updates_list = []
    worker_id_list = []
    agent_updates_dict = {}
    mask_aggrement = []

    acc_vec = []
    asr_vec = []
    pacc_vec = []
    per_class_vec = []

    clean_asr_vec = []
    clean_acc_vec = []
    clean_pacc_vec = []
    clean_per_class_vec = []

    for rnd in tqdm(range(1, args.rounds + 1)):

        logging.info("--------round {} ------------".format(rnd))
        print("--------round {} ------------".format(rnd))
        # mask = torch.ones(n_model_params)
        rnd_global_params = parameters_to_vector([ copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()])

        # agent_updates_dict_prev = copy.deepcopy(agent_updates_dict)
        agent_updates_dict = {}
        chosen = np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False)
        if args.method == "lockdown" or args.method == "fedimp":
            old_mask = [copy.deepcopy(agent.mask) for agent in agents]
        for agent_id in chosen:
            # logging.info(torch.sum(rnd_global_params))
            global_model = global_model.to(args.device)
            if args.method == "lockdown" or args.method == "fedimp":
                update = agents[agent_id].local_train(global_model, criterion, rnd, global_mask=global_mask, neurotoxin_mask = neurotoxin_mask, updates_dict=updates_dict)
            else:
                update = agents[agent_id].local_train(global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask)
            agent_updates_dict[agent_id] = update
            utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

        # aggregate params obtained by agents and update the global params
        if args.method == "lockdown" or args.method == "fedimp":
            _, _, updates_dict,neurotoxin_mask = aggregator.aggregate_updates(global_model, agent_updates_dict, rnd, masks=old_mask,
                                                   agent_updates_dict_prev=None,
                                                   mask_aggrement=mask_aggrement)
        else:
            _, _, updates_dict,neurotoxin_mask = aggregator.aggregate_updates(global_model, agent_updates_dict, rnd,
                                                   agent_updates_dict_prev=None,
                                                   mask_aggrement=mask_aggrement)
        # agent_updates_list.append(np.array(server_update.to("cpu")))
        worker_id_list.append(agent_id + 1)



        # inference in every args.snap rounds
        if rnd % args.snap == 0:
            val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader,
                                                                                  args, rnd, num_target)
            # writer.add_scalar('Validation/Loss', val_loss, rnd)
            # writer.add_scalar('Validation/Accuracy', val_acc, rnd)
            logging.info(f'| Val_Acc:   {val_acc:.3f} |')
            logging.info(f'| Val_Per_Class_Acc: {val_per_class_acc} ')
            acc_vec.append(val_acc)
            per_class_vec.append(val_per_class_acc)

            poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                                                                            poisoned_val_loader, args, rnd, num_target)
            cum_poison_acc_mean += asr
            asr_vec.append(asr)
            logging.info(f'| Attack Success Ratio: {asr:.3f} |')

            poison_loss, (poison_acc, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion,
                                                                                   poisoned_val_only_x_loader, args,
                                                                                   rnd, num_target)
            pacc_vec.append(poison_acc)
            logging.info(f'| Poison Loss/Poison accuracy: {poison_loss:.3f} / {poison_acc:.3f} |')

            # plt.imshow(np.transpose(fail_samples[0].to("cpu"), (1, 2, 0)))
            # plt.show()
            # plt.imshow(np.transpose(fail_samples[1].to("cpu"), (1, 2, 0)))
            # plt.show()
            # plt.imshow(np.transpose(fail_samples[2].to("cpu"), (1, 2, 0)))
            # plt.show()
            if args.method == "lockdown" or args.method == "fedimp":
                test_model = copy.deepcopy(global_model)
                for name, param in test_model.named_parameters():
                    mask = 0
                    for id, agent in enumerate(agents):
                        mask += old_mask[id][name].to(args.device)
                    param.data = torch.where(mask.to(args.device) >= args.theta, param,
                                             torch.zeros_like(param))
                    logging.info(torch.sum(mask.to(args.device) >= args.theta) / torch.numel(mask))
                val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                      val_loader,
                                                                                      args, rnd, num_target)
                # writer.add_scalar('Clean Validation/Loss', val_loss, rnd)
                # writer.add_scalar('Clean Validation/Accuracy', val_acc, rnd)
                logging.info(f'| Clean Val_Acc:   {val_acc:.3f} |')
                logging.info(f'| Clean Val_Per_Class_Acc: {val_per_class_acc} ')
                clean_acc_vec.append(val_acc)
                clean_per_class_vec.append(val_per_class_acc)

                poison_loss, (poison_acc, _), _ = utils.get_loss_n_accuracy(test_model, criterion,
                                                                            poisoned_val_loader,
                                                                            args, rnd, num_target)
                clean_asr_vec.append(poison_acc)
                cum_poison_acc_mean += poison_acc
                # writer.add_scalar('Clean Poison/Base_Class_Accuracy', val_per_class_acc[args.base_class], rnd)
                # writer.add_scalar('Clean Poison/Poison_Accuracy', poison_acc, rnd)
                # writer.add_scalar('Clean Poison/Poison_Loss', poison_loss, rnd)
                # writer.add_scalar('Clean Poison/Cumulative_Poison_Accuracy_Mean', cum_poison_acc_mean / rnd, rnd)
                logging.info(f'| Clean Attack Success Ratio: {poison_loss:.3f} / {poison_acc:.3f} |')

                poison_loss, (poison_acc, _), fail_samples = utils.get_loss_n_accuracy(test_model, criterion,
                                                                                       poisoned_val_only_x_loader, args,
                                                                                       rnd, num_target)
                clean_pacc_vec.append(poison_acc)
                logging.info(f'| Clean Poison Loss/Clean Poison accuracy: {poison_loss:.3f} / {poison_acc:.3f} |')
                # ask the guys to finetune the classifier
                del test_model
        if args.data != "fedemnist":
            save_frequency = 25
        else:
            save_frequency = 100
        if rnd % save_frequency == 0:
            if args.mask_init == "uniform":
                PATH = "checkpoint/uniform_AckRatio{}_{}_Method{}_data{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_se_threshold{}_noniid{}_maskthreshold{}_attack{}.pt".format(
                    args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, rnd, args.local_ep,
                    args.poison_frac, args.dense_ratio, args.aggr, args.se_threshold, args.non_iid,
                    args.theta, args.attack)
            elif args.dis_check_gradient == True:
                PATH = "checkpoint/NoGradient_AckRatio{}_{}_Method{}_data{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_se_threshold{}_noniid{}_maskthreshold{}_attack{}.pt".format(
                    args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, rnd, args.local_ep,
                    args.poison_frac, args.dense_ratio, args.aggr, args.se_threshold, args.non_iid,
                    args.theta, args.attack)
            else:
                PATH = "checkpoint/AckRatio{}_{}_Method{}_data{}_alpha{}_Rnd{}_Epoch{}_inject{}_dense{}_Agg{}_se_threshold{}_noniid{}_maskthreshold{}_attack{}_prox0.1.pt".format(
                    args.num_corrupt, args.num_agents, args.method, args.data, args.alpha, rnd, args.local_ep,
                    args.poison_frac, args.dense_ratio, args.aggr, args.se_threshold, args.non_iid,
                    args.theta, args.attack)
            if args.method == "lockdown" or args.method == "fedimp":
                torch.save({
                    'option': args,
                    'model_state_dict': global_model.state_dict(),
                    'masks': [agent.mask for agent in agents],
                    'acc_vec': acc_vec,
                    "asr_vec": asr_vec,
                    'pacc_vec ': pacc_vec,
                    "per_class_vec": per_class_vec,
                    "clean_asr_vec": clean_asr_vec,
                    'clean_acc_vec': clean_acc_vec,
                    'clean_pacc_vec ': clean_pacc_vec,
                    'clean_per_class_vec': clean_per_class_vec,
                }, PATH)
            else:
                torch.save({
                    'option': args,
                    'model_state_dict': global_model.state_dict(),
                    'acc_vec': acc_vec,
                    "asr_vec": asr_vec,
                    'pacc_vec ': pacc_vec,
                    "per_class_vec": per_class_vec,
                    'neurotoxin_mask': neurotoxin_mask
                }, PATH)

    # logging.info(mask_aggrement)
    rnd = 0
    logging.info('Training has finished!')

    if args.rounds > 0:
        exit()
    elif args.rounds < 0:
        if args.rounds != -1:
            combined_dataset = ConcatDataset([agents[agent_id].train_loader.dataset for agent_id in range(args.num_agents)])
            # Create a single DataLoader
            args.combined_train_loader = DataLoader(
                # train_dataset_copy,
                combined_dataset,
                batch_size=args.bs,  # Make sure 'args.bs' is defined and accessible
                shuffle=True,
                num_workers=args.num_workers,  # Make sure 'args.num_workers' is defined and accessible
                pin_memory=False,
                drop_last=True
            )   
            args.val_loader = val_loader
            args.poisoned_val_loader = poisoned_val_loader
            global_model = global_train(args, global_model, criterion, round=200)
            PATH = "checkpoint/combined_train_prox.pt"
            torch.save({'model_state_dict': global_model.state_dict()}, PATH)
            exit()
        else:
             for rnd in tqdm(range(1)):
                logging.info("--------round {} ------------".format(rnd))
                print("--------round {} ------------".format(rnd))
                # mask = torch.ones(n_model_params)
                rnd_global_params = parameters_to_vector([ copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()])

                # agent_updates_dict_prev = copy.deepcopy(agent_updates_dict)
                agent_updates_dict = {}
                chosen = np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False)
                if args.method == "lockdown" or args.method == "fedimp":
                    old_mask = [copy.deepcopy(agent.mask) for agent in agents]
                for agent_id in chosen:
                    # logging.info(torch.sum(rnd_global_params))
                    global_model = global_model.to(args.device)
                    if args.method == "lockdown" or args.method == "fedimp":
                        update = agents[agent_id].local_train(global_model, criterion, rnd, global_mask=global_mask, neurotoxin_mask = neurotoxin_mask, updates_dict=updates_dict)
                    else:
                        update = agents[agent_id].local_train(global_model, criterion, rnd, neurotoxin_mask=neurotoxin_mask)
                    agent_updates_dict[agent_id] = update
                    if agent_id == args.num_agents - 1:
                        args.val_frac = 0.01
                        args.clean_label = -1
                        args.print_every = 500
                        args.batch_size = 128
                        _, mask_values =  train_mask(-1, global_model, criterion, server_train_loader, mask_lr, anp_eps, anp_steps, anp_alpha, round)
                        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
                        print(f'mask_values:{mask_values[0]} - {mask_values[100]} - {mask_values[1000]}')
                        local_model = copy.deepcopy(global_model)
                        print('-' * 64)
                        print(f'|settings: {mask_lr}, {anp_eps}, {anp_steps}, {anp_alpha}, {round} |')
                        results = []
                        thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
                        start = 0
                        print('pruning')
                        for threshold in thresholds:
                            print('-' * 64)
                            idx = start
                            for idx in range(start, len(mask_values)):
                                if float(mask_values[idx][2]) <= threshold:
                                    pruning(local_model, mask_values[idx])
                                    start += 1
                                else:
                                    break
                            layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
                            print(f'layer_name:{layer_name}, neuron_idx:{neuron_idx}, value:{value}')
                            with torch.no_grad():
                                val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(local_model, criterion, val_loader, args, rnd, num_target)
                                print(f'| Val_Acc:   {val_acc:.3f} |')
                                acc_vec.append(val_acc)
                                per_class_vec.append(val_per_class_acc)
                                poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(local_model, criterion, poisoned_val_loader, args, rnd, num_target)
                                cum_poison_acc_mean += asr
                                asr_vec.append(asr)
                                print(f'| Attack Success Ratio: {asr:.3f} |')
                        exit()
                    utils.vector_to_model(copy.deepcopy(rnd_global_params), global_model)

    else:
        with torch.no_grad():
            val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args, rnd, num_target)
            print(f'| Val_Acc:   {val_acc:.3f} |')
            acc_vec.append(val_acc)
            per_class_vec.append(val_per_class_acc)
            poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args, rnd, num_target)
            cum_poison_acc_mean += asr
            asr_vec.append(asr)
            print(f'| Attack Success Ratio: {asr:.3f} |')

        args.dataset = 'CIFAR10'
        args.clean_threshold = 0.25
        args.unlearning_lr = 0.01
        args.unlearning_epochs = 20
        args.schedule = [10, 20]
        args.ratio = 0.01
        args.batch_size = 128
        defense_data_loader = get_train_loader(args) 
        criterion = torch.nn.CrossEntropyLoss().to(args.device)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.unlearning_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

        logging.info('----------- Model Unlearning --------------')
        logging.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        for epoch in range(0, args.unlearning_epochs + 1):
            start = time.time()
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc = train_step_unlearning(args=args, model=global_model, criterion=criterion, optimizer=optimizer,
                                        data_loader=defense_data_loader)
            scheduler.step()
            end = time.time()
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args, rnd, num_target)
                # print(f'| Val_Acc:   {val_acc:.3f} |')
                acc_vec.append(val_acc)
                per_class_vec.append(val_per_class_acc)
                poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args, rnd, num_target)
                cum_poison_acc_mean += asr
                asr_vec.append(asr)
                # print(f'| Attack Success Ratio: {asr:.3f} |')
            logging.info(
                '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                epoch, lr, end-start, train_loss, train_acc, poison_loss, asr,
                val_loss, val_acc
            )

            if train_acc <= args.clean_threshold:
                # save the last checkpoint
                file_path = os.path.join('checkpoint/unlearned_model_last.tar')
                # torch.save(net.state_dict(), os.path.join(args.output_dir, 'unlearned_model_last.tar'))
                torch.save({
                    'epoch': epoch,
                    'state_dict': global_model.state_dict(),
                    'clean_acc': val_acc,
                    'bad_acc': asr,
                    'optimizer': optimizer.state_dict(),
                }, file_path)
                break
        
        logging.info('----------- Model Recovering --------------')
        # Step 2: load unleanred model checkpoints
        args.unlearned_model_path = 'checkpoint/unlearned_model_last.tar'
        args.arch = 'resnet18'
        args.recovering_lr = 0.2
        args.recovering_epochs = 20
        device = 'cuda:0'
        if args.unlearned_model_path is not None:
            unlearned_model_path = args.unlearned_model_path
        else:
            unlearned_model_path = os.path.join(args.output_weight, 'unlearned_model_last.tar')

        checkpoint = torch.load(unlearned_model_path, map_location=device)
        print('Unlearned Model:', checkpoint['epoch'], checkpoint['clean_acc'], checkpoint['bad_acc'])

        unlearned_model = models.get_model(args.data).to(args.device)
        print('replace batchnorm')
        replace_batchnorm_with_noisy(unlearned_model)
        print('replace batchnorm done')
        load_state_dict(unlearned_model, orig_state_dict=checkpoint['state_dict'])
        unlearned_model = unlearned_model.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)

        parameters = list(unlearned_model.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=args.recovering_lr, momentum=0.9)

        # Recovering
        logging.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        for epoch in range(1, args.recovering_epochs + 1):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            train_loss, train_acc = train_step_recovering(args=args, unlearned_model=unlearned_model, criterion=criterion, data_loader=defense_data_loader,
                                            mask_opt=mask_optimizer)
            end = time.time()
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(global_model, criterion, val_loader, args, rnd, num_target)
                # print(f'| Val_Acc:   {val_acc:.3f} |')
                acc_vec.append(val_acc)
                per_class_vec.append(val_per_class_acc)
                poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(global_model, criterion, poisoned_val_loader, args, rnd, num_target)
                cum_poison_acc_mean += asr
                asr_vec.append(asr)
                # print(f'| Attack Success Ratio: {asr:.3f} |')
            logging.info(
                '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                epoch, lr, end-start, train_loss, train_acc, poison_loss, asr,
                val_loss, val_acc
            )
        save_mask_scores(unlearned_model.state_dict(), os.path.join(args.log_root, 'mask_values.txt'))
        del unlearned_model


        exit()
























    args.val_frac = 0.01
    args.clean_label = -1
    args.print_every = 500
    args.batch_size = 128
    _, clean_val = poison.split_dataset(dataset=train_dataset, val_frac=args.val_frac,
                                        perm=np.loadtxt('./cifar_shuffle.txt', dtype=int), clean_label = args.clean_label)
    random_sampler = RandomSampler(data_source=clean_val, replacement=True,
                            num_samples=args.print_every * args.batch_size)
    server_train_loader = DataLoader(clean_val, batch_size=args.batch_size,
                            shuffle=False, sampler=random_sampler, num_workers=0)
    best_val_acc = 0
    best_asr = 1
    id2mask_values = {}
    pruning_max, pruning_step = 0.95, 0.05
    best_diff, best_diff_ = 0, 0

    for _ in range(1):
        for mask_lr in [0.1]:
            for anp_eps in [1.0]:
                for anp_steps in [1]:
                    for anp_alpha in [0.2]:
                        for round in [10]:
                            _, mask_values =  train_mask(-1, global_model, criterion, server_train_loader, mask_lr, anp_eps, anp_steps, anp_alpha, round)
                            id2mask_values[-1] = torch.tensor([mask_values[i][-1] for i in range(len(mask_values))])
                            mask_values = sorted(mask_values, key=lambda x: float(x[2]))
                            print(f'mask_values:{mask_values[0]} - {mask_values[100]} - {mask_values[1000]}')
                            local_model = copy.deepcopy(global_model)
                            print('-' * 64)
                            print(f'|settings: {mask_lr}, {anp_eps}, {anp_steps}, {anp_alpha}, {round} |')
                            results = []
                            thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
                            start = 0
                            print('pruning')
                            for threshold in thresholds:
                                print('-' * 64)
                                idx = start
                                for idx in range(start, len(mask_values)):
                                    if float(mask_values[idx][2]) <= threshold:
                                        pruning(local_model, mask_values[idx])
                                        start += 1
                                    else:
                                        break
                                layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
                                print(f'layer_name:{layer_name}, neuron_idx:{neuron_idx}, value:{value}')
                                with torch.no_grad():
                                    val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(local_model, criterion, val_loader, args, rnd, num_target)
                                    print(f'| Val_Acc:   {val_acc:.3f} |')
                                    acc_vec.append(val_acc)
                                    per_class_vec.append(val_per_class_acc)
                                    poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(local_model, criterion, poisoned_val_loader, args, rnd, num_target)
                                    cum_poison_acc_mean += asr
                                    asr_vec.append(asr)
                                    print(f'| Attack Success Ratio: {asr:.3f} |')
                                    if val_acc - asr > best_diff:
                                        best_val_acc = val_acc
                                        best_asr = asr
                                        best_diff = val_acc - asr
                                        best_diff_ = f'{mask_lr}, {anp_eps}, {anp_steps}, {anp_alpha}, {round}'
                                if False:
                                    print('finetune')
                                    args.combined_train_loader = server_train_loader
                                    args.client_lr = 0.001
                                    local_model = global_train(args, local_model, criterion, round=1)
                                    with torch.no_grad():
                                        val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(local_model, criterion, val_loader, args, rnd, num_target)
                                        print(f'| Val_Acc:   {val_acc:.3f} |')
                                        acc_vec.append(val_acc)
                                        per_class_vec.append(val_per_class_acc)
                                        poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(local_model, criterion, poisoned_val_loader, args, rnd, num_target)
                                        cum_poison_acc_mean += asr
                                        asr_vec.append(asr)
                                        print(f'| Attack Success Ratio: {asr:.3f} |')
                                    print('finetune done')

        print(f'best_val_acc:{best_val_acc}')
        print(f'best_asr:{best_asr}')
        print(f'best_diff:{best_diff}, best_diff_:{best_diff_}')



    mask_lr, anp_eps, anp_steps, anp_alpha, round = 0.1, 1.0, 1, 0.2, 10
    cos_matrix = {}
    for rnd in tqdm(range(1, 1)):
        rnd_global_params = parameters_to_vector([ copy.deepcopy(global_model.state_dict()[name]) for name in global_model.state_dict()])
        agent_updates_dict = {}
        chosen = np.random.choice(args.num_agents, math.floor(args.num_agents * args.agent_frac), replace=False)

        # chosen = [0, 20]
        for agent_id in chosen:
            print('-' * 64)
            global_model = global_model.to(args.device)
            local_model, mask_values =  train_mask(agent_id, global_model, criterion, agents[agent_id].train_loader, mask_lr, anp_eps, anp_steps, anp_alpha, round)
            id2mask_values[agent_id] = torch.tensor([[mask_values[i][-1] for i in range(len(mask_values)) if i > len(mask_values) //2]])
            print(f'|settings: {mask_lr}, {anp_eps}, {anp_steps}, {anp_alpha}, {round} |')
            with torch.no_grad():
                val_loss, (val_acc, val_per_class_acc), _ = utils.get_loss_n_accuracy(local_model, criterion, val_loader,args, rnd, num_target)
                print(f'| Val_Acc:   {val_acc:.3f} |')
                acc_vec.append(val_acc)
                per_class_vec.append(val_per_class_acc)
                poison_loss, (asr, _), fail_samples = utils.get_loss_n_accuracy(local_model, criterion, poisoned_val_loader, args, rnd, num_target)
                cum_poison_acc_mean += asr
                asr_vec.append(asr)
                print(f'| Attack Success Ratio: {asr:.3f} |')

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for agent_id in chosen:
            print(id2mask_values[agent_id].shape)
            print(id2mask_values[-1].shape)
            cos_matrix[agent_id] = cos(id2mask_values[agent_id], id2mask_values[-1])
        print(cos_matrix)
            