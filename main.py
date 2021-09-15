import argparse
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.backends import cudnn

from network import init_weights, Vanilla_U_Net, R2AttU_Net
from utils import *


def main(config):
    cudnn.benchmark = True
    if config.model_type not in ['Vanilla_U_Net','R2AttU_Net']:
        print('ERROR!! model_type should be selected in Vanilla_U_Net/R2AttU_Net')
        print('Your input for model_type was %s'%config.model_type)
        return

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)


    # data
    train_X = np.load(config.train_path+'train_X_256.npy')  # [:, 0, :, :].reshape(-1, 1, 256, 256)  # amplitude only
    valid_X = np.load(config.valid_path+'valid_X_256.npy')  # [:, 0, :, :].reshape(-1, 1, 256, 256)  # amplitude only
    # test_X = np.load(config.test_path+'test_X_256.npy')  # [:, 0, :, :].reshape(-1, 1, 256, 256)  # amplitude only

    train_y = np.load(config.train_path+'train_y_256.npy')
    valid_y = np.load(config.train_path+'valid_y_256.npy')
    # test_y = np.load(config.test_path+'test_y_256.npy')

    train_dataset = TensorDataset(torch.tensor(train_X), torch.tensor(train_y))
    valid_dataset = TensorDataset(torch.tensor(valid_X), torch.tensor(valid_y))
    # test_dataset = TensorDataset(torch.tensor(test_X), torch.tensor(test_y))

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.barch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=config.barch_size, shuffle=True, num_workers=config.num_workers)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=config.barch_size, shuffle=True, num_workers=config.num_workers)


    # setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    # CUDA_LAUNCH_BLOCKING=1

    lr = config.lr
    beta1 = config.beta1
    beta2 = config.beta2

    num_epochs = config.num_epochs
    num_decay_epochs = config.num_decay_epochs

    Net = R2AttU_Net(img_ch=config.img_ch, output_ch=config.output_ch, t=config.t)  # Vanilla_U_Net(img_ch=config.img_ch, output_ch=config.output_ch)
    Net.to(device)
    init_weights(Net, init_type='xavier', gain=config.init_gain)

    Loss = torch.nn.BCELoss()
    Optimizer = optim.Adam(list(Net.parameters()), lr, [beta1, beta2])


    # train
    epoch_losses = []
    best_score = 0.

    for i in range(num_epochs):
        print('=' * 7, 'training starts', '=' * 7)

        Net.train(True)

        epoch_loss = 0.

        acc = 0.
        SE = 0.
        SP = 0.
        PC = 0.
        F1 = 0.
        JS = 0.
        DC = 0.
        count = 0

        for j, (img, truth) in enumerate(train_loader):
            img = img.float()  # [:,0,:,:].view(2,-1,256,256)
            truth = truth.float()

            img = img.to(device)
            truth = truth.to(device)

            pred = Net.forward(img)
            pred_prob = F.sigmoid(pred)

            loss = Loss(pred_prob.view(pred_prob.size(0), -1),
                        truth.view(truth.size(0), -1))
            epoch_loss += loss.item()

            # back-propagation
            Net.zero_grad()
            loss.backward()
            Optimizer.step()

            acc += get_accuracy(pred_prob, truth)
            SE += get_sensitivity(pred_prob, truth)
            SP += get_specificity(pred_prob, truth)
            PC += get_precision(pred_prob, truth)
            F1 += get_F1(pred_prob, truth)
            JS += get_JS(pred_prob, truth)
            DC += get_DC(pred_prob, truth)
            count += img.size(0)

        torch.cuda.empty_cache()

        acc /= count
        SE /= count
        SP /= count
        PC /= count
        F1 /= count
        JS /= count
        DC /= count

        epoch_losses.append(epoch_loss)
        print(
            'Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'
            % (i + 1, num_epochs, epoch_loss, acc, SE, SP, PC, F1, JS, DC))

        if (i + 1) > num_decay_epochs:
            lr -= (lr / float(num_decay_epochs))
            for param_group in Optimizer.param_groups:
                param_group['lr'] = lr


        # validation
        torch.cuda.empty_cache()
        print('=' * 7, 'validation starts', '=' * 7)

        Net.train(False)
        Net.eval()

        acc = 0.
        SE = 0.
        SP = 0.
        PC = 0.
        F1 = 0.
        JS = 0.
        DC = 0.
        count = 0

        # with torch.no_grad():
        for k, (img, truth) in enumerate(valid_loader):
            img = img.float()  # [:,0,:,:].view(2,-1,256,256)
            truth = truth.float()

            img = img.to(device)
            truth = truth.to(device)

            pred = Net.forward(img)
            pred_prob = F.sigmoid(pred)

            acc += get_accuracy(pred_prob, truth)
            SE += get_sensitivity(pred_prob, truth)
            SP += get_specificity(pred_prob, truth)
            PC += get_precision(pred_prob, truth)
            F1 += get_F1(pred_prob, truth)
            JS += get_JS(pred_prob, truth)
            DC += get_DC(pred_prob, truth)
            count += img.size(0)

        acc /= count
        SE /= count
        SP /= count
        PC /= count
        F1 /= count
        JS /= count
        DC /= count

        score = JS+DC  # acc

        print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
        acc, SE, SP, PC, F1, JS, DC))

        if score > best_score and i > 0.7 * num_epochs:
            best_score = score

            best_epoch = i
            best_net = Net.state_dict()

            print('Autobot at epoch {}!'.format(best_epoch))
            torch.save(best_net, 'model.pth')

        torch.cuda.empty_cache()


    plt.plot(list(range(len(epoch_losses))), epoch_losses)
    plt.title('Results')
    plt.xlabel('epoch')
    plt.xlabel('loss')
    plt.savefig('results2')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyper-parameters
    parser.add_argument('--image_size', type=int, default=512)

    parser.add_argument('--t', type=int, default=2, help='t for recurrent step number')
    parser.add_argument('--img_ch', type=int, default=2)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--init_gain', type=float, delfault=0.02)

    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--num_decay_epochs', type=int, default=75)

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)

    parser.add_argument('--model_type', type=str, default='R2AttU_Net', help='Vanilla_U_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--train_path', type=str, default='./data/train/')
    parser.add_argument('--valid_path', type=str, default='./data/valid/')
    parser.add_argument('--test_path', type=str, default='./data/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    config = parser.parse_args()

    main(config)
