import os
import time
import argparse
import numpy as np
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from anp_batchnorm import NoisyBatchNorm2d, NoisyBatchNorm1d
import torch.nn as nn

from prune_neuron_cifar import read_data
from prune_neuron_cifar import prune_by_threshold
# import data.poison_cifar as poison

# parser = argparse.ArgumentParser(description='Train poisoned networks')

# # Basic model parameters.
# parser.add_argument('--arch', type=str, default='resnet18',
#                     choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2', 'vgg19_bn'])
# parser.add_argument('--checkpoint', type=str, required=True, help='The checkpoint to be pruned')
# parser.add_argument('--widen-factor', type=int, default=1, help='widen_factor for WideResNet')
# parser.add_argument('--batch-size', type=int, default=128, help='the batch size for dataloader')
# parser.add_argument('--lr', type=float, default=0.2, help='the learning rate for mask optimization')
# parser.add_argument('--nb-iter', type=int, default=2000, help='the number of iterations for training')
# parser.add_argument('--print-every', type=int, default=500, help='print results every few iterations')
# parser.add_argument('--data-dir', type=str, default='../data', help='dir to the dataset')
# parser.add_argument('--val-frac', type=float, default=0.01, help='The fraction of the validate set')
# parser.add_argument('--output-dir', type=str, default='logs/models/')

# parser.add_argument('--trigger-info', type=str, default='', help='The information of backdoor trigger')
# parser.add_argument('--poison-type', type=str, default='benign', choices=['badnets', 'blend', 'clean-label', 'benign'],
#                     help='type of backdoor attacks for evaluation')
# parser.add_argument('--poison-target', type=int, default=0, help='target class of backdoor attack')
# parser.add_argument('--trigger-alpha', type=float, default=1.0, help='the transparency of the trigger pattern.')

# parser.add_argument('--anp-eps', type=float, default=0.4)
# parser.add_argument('--anp-steps', type=int, default=1)
# parser.add_argument('--anp-alpha', type=float, default=0.2)

args = {}
# args_dict = vars(args)
# print(args_dict)
# os.makedirs(args.output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# def main():
#     MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
#     STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
#     ])
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
#     ])

#     # Step 1: create dataset - clean val set, poisoned test set, and clean test set.
#     if args.trigger_info:
#         trigger_info = torch.load(args.trigger_info, map_location=device)
#     else:
#         if args.poison_type == 'benign':
#             trigger_info = None
#         else:
#             triggers = {'badnets': 'checkerboard_1corner',
#                         'clean-label': 'checkerboard_4corner',
#                         'blend': 'gaussian_noise'}
#             trigger_type = triggers[args.poison_type]
#             pattern, mask = poison.generate_trigger(trigger_type=trigger_type)
#             trigger_info = {'trigger_pattern': pattern[np.newaxis, :, :, :], 'trigger_mask': mask[np.newaxis, :, :, :],
#                             'trigger_alpha': args.trigger_alpha, 'poison_target': np.array([args.poison_target])}

#     orig_train = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
#     _, clean_val = poison.split_dataset(dataset=orig_train, val_frac=args.val_frac,
#                                         perm=np.loadtxt('./data/cifar_shuffle.txt', dtype=int))
#     clean_test = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
#     poison_test = poison.add_predefined_trigger_cifar(data_set=clean_test, trigger_info=trigger_info)

#     random_sampler = RandomSampler(data_source=clean_val, replacement=True,
#                                    num_samples=args.print_every * args.batch_size)
#     clean_val_loader = DataLoader(clean_val, batch_size=args.batch_size,
#                                   shuffle=False, sampler=random_sampler, num_workers=0)
#     poison_test_loader = DataLoader(poison_test, batch_size=args.batch_size, num_workers=0)
#     clean_test_loader = DataLoader(clean_test, batch_size=args.batch_size, num_workers=0)

#     # Step 2: load model checkpoints and trigger info
#     state_dict = torch.load(args.checkpoint, map_location=device)
#     net = getattr(models, args.arch)(num_classes=10, norm_layer=NoisyBatchNorm2d)
#     load_state_dict(net, orig_state_dict=state_dict)
#     net = net.to(device)
#     criterion = torch.nn.CrossEntropyLoss().to(device)

#     parameters = list(net.named_parameters())
#     mask_params = [v for n, v in parameters if "neuron_mask" in n]
#     mask_optimizer = torch.optim.SGD(mask_params, lr=args.lr, momentum=0.9)
#     noise_params = [v for n, v in parameters if "neuron_noise" in n]
#     noise_optimizer = torch.optim.SGD(noise_params, lr=args.anp_eps / args.anp_steps)

#     # Step 3: train backdoored models
#     print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
#     nb_repeat = int(np.ceil(args.nb_iter / args.print_every))
#     for i in range(nb_repeat):
#         start = time.time()
#         lr = mask_optimizer.param_groups[0]['lr']
#         train_loss, train_acc = mask_train(model=net, criterion=criterion, data_loader=clean_val_loader,
#                                            mask_opt=mask_optimizer, noise_opt=noise_optimizer)
#         cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
#         po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
#         end = time.time()
#         print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
#             (i + 1) * args.print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
#             cl_test_loss, cl_test_acc))
#     save_mask_scores(net.state_dict(), os.path.join(args.output_dir, 'mask_values.txt'))

def replace_bn_with_noisy_bn(module: nn.Module) -> nn.Module:
    """Recursively replace all BatchNorm layers with NoisyBatchNorm layers while preserving weights."""
    device = 'cuda:0'
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Create a new NoisyBatchNorm2d layer
            new_layer = NoisyBatchNorm2d(child.num_features).to(device=device)
            
            # Copy weights and biases
            new_layer.weight.data = child.weight.data.clone().detach()
            new_layer.bias.data = child.bias.data.clone().detach()
            
            # Copy running mean and variance
            new_layer.running_mean = child.running_mean.clone().detach()
            new_layer.running_var = child.running_var.clone().detach()
            
            # Replace the original layer with the new layer
            setattr(module, name, new_layer)
        elif isinstance(child, nn.BatchNorm1d):
            # Create a new NoisyBatchNorm1d layer
            new_layer = NoisyBatchNorm1d(child.num_features).to(device=device)
            
            # Copy weights and biases
            new_layer.weight.data = child.weight.data.clone().detach()
            new_layer.bias.data = child.bias.data.clone().detach()
            
            # Copy running mean and variance
            new_layer.running_mean = child.running_mean.clone().detach()
            new_layer.running_var = child.running_var.clone().detach()
            
            # Replace the original layer with the new layer
            setattr(module, name, new_layer)
        else:
            replace_bn_with_noisy_bn(child)
    return module

def train_mask(id, global_model, criterion, train_loader, mask_lr, anp_eps, anp_steps, anp_alpha, round):
        print(f'id:{id}')
        device = 'cuda:0'
        from copy import deepcopy   
        local_model = deepcopy(global_model)
        local_model = replace_bn_with_noisy_bn(local_model)
        local_model.train()
        local_model = local_model.to(device)
        local_model.mask_lr = mask_lr
        local_model.anp_eps = anp_eps
        local_model.anp_steps = anp_steps
        local_model.anp_alpha = anp_alpha
        mask_scores = None

        local_model.train()  
        parameters = list(local_model.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=local_model.mask_lr, momentum=0.9)
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=local_model.anp_eps / local_model.anp_steps)

        for epoch in range(round):
            train_loss, train_acc = mask_train(model=local_model, criterion=criterion, data_loader=train_loader,
                                        mask_opt=mask_optimizer, noise_opt=noise_optimizer)
        mask_scores = get_mask_scores(local_model.state_dict())
        save_mask_scores(local_model.state_dict(), f'/work/LAS/wzhang-lab/mingl/code/backdoor/Defending-Against-Backdoors-with-Robust-Learning-Rate/save/mask_values{id}.txt')
        mask_values = read_data(f'/work/LAS/wzhang-lab/mingl/code/backdoor/Defending-Against-Backdoors-with-Robust-Learning-Rate/save/mask_values{id}.txt')
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        print(f'mask_values:{mask_values[0]} - {mask_values[100]} - {mask_values[1000]}')
        prune_by_threshold(global_model, mask_values, pruning_max=0.6, pruning_step=0.05)
        return local_model, mask_scores

def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, NoisyBatchNorm2d) or isinstance(module, NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=model.anp_eps)


def mask_train(model, criterion, mask_opt, noise_opt, data_loader):
    # is_malicious = model.is_malicious
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    max_nb_samples = 1000
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        # step 1: calculate the adversarial perturbation for neurons
        if model.anp_eps > 0.0:
            reset(model, rand_init=True)
            for _ in range(model.anp_steps):
                noise_opt.zero_grad()

                include_noise(model)
                output_noise = model(images)
                loss_noise = - criterion(output_noise, labels)

                loss_noise.backward()
                sign_grad(model)
                noise_opt.step()

        # step 2: calculate loss and update the mask values
        mask_opt.zero_grad()
        if model.anp_eps > 0.0:
            include_noise(model)
            output_noise = model(images)
            loss_rob = criterion(output_noise, labels)
        else:
            loss_rob = 0.0

        exclude_noise(model)
        output_clean = model(images)
        loss_nat = criterion(output_clean, labels)

        loss = model.anp_alpha * loss_nat + (1 - model.anp_alpha) * loss_rob

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(model)
        # print(f'loss:{loss} loss_nat:{loss_nat} loss_rob:{loss_rob}')
        if nb_samples > max_nb_samples:
            break

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
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

def get_mask_scores(state_dict):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    return mask_values



# if __name__ == '__main__':
    # main()
