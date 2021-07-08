import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

from adversarialbox.attacks import FGSMAttack, LinfPGDAttack
from adversarialbox.train import adv_train, FGSM_train_rnd
from adversarialbox.utils import to_var, pred_batch, test
from time import time
from models import LeNet5
from adversarialbox.utils import to_var, pred_batch, test, \
    attack_over_test_data

import math

class _Quantize(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)

        N_bits = 6
        ttanh = torch.tanh(input)
        #abss = torch.abs(ttanh)
        #maxx = torch.max(abss)
        #Func = ttanh / (maxx * 2) + 0.5
        output = (ttanh * (2 ** N_bits - 1)).round() / (2 ** N_bits - 1)
        #output = Quant_0_1 * 2 - 1
        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class _BinActive(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        output = torch.sign(input)
        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        grad_input = grad_output.clone()
        input, = self.saved_tensors
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class _TernActive(torch.autograd.Function):

    def forward(self, input):
        self.save_for_backward(input)
        E_w = input.abs().max()
        self.th = 0.05*E_w #threshold
        output = input.clone().zero_()
        self.W = input[input.ge(self.th)+input.le(-self.th)].abs().mean()
        output[input.ge(self.th)] = self.W
        output[input.le(-self.th)] = -self.W
        return output

    def backward(self, grad_output):
        # saved tensors - tuple of tensors with one element
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class bilinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features)
    def forward(self, input):
        bw = _TernActive()(self.weight)
        bb = _TernActive()(self.bias)
        return F.linear(input, bw, bb)

class quantized_conv(nn.Conv2d):
    # def __init__(self,nchin,nchout,sz,stride,padding):
    #     super().__init__(in_channels=nchin,out_channels=nchout, kernel_size=sz, padding=padding, stride=stride, bias=False)

    def forward(self, input):
        qw = _TernActive()(self.weight)
        # qw = _BinActive()(self.weight)
        # qw = torch.tanh(self.weight)
        return F.conv2d(input, qw, self.bias, self.stride, self.padding, self.dilation, self.groups)

 

# Hyper-parameters
param = {
    'batch_size': 256,
    'test_batch_size': 128,
    'num_epochs': 40,
    'delay': 20,
    'learning_rate': 1e-3,
    'weight_decay': 5e-5,
}




# Data loaders
train_dataset = datasets.MNIST(root='../data/',train=True, download=True, 
    transform=transforms.ToTensor())
loader_train = torch.utils.data.DataLoader(train_dataset, 
    batch_size=param['batch_size'], shuffle=True)

test_dataset = datasets.MNIST(root='../data/', train=False, download=True, 
    transform=transforms.ToTensor())
loader_test = torch.utils.data.DataLoader(test_dataset, 
    batch_size=param['test_batch_size'], shuffle=True)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = quantized_conv(1, 32, kernel_size=3, padding=1, stride=1)
        #self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = quantized_conv(32, 64, kernel_size=3, padding=1, stride=1)
        #self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = bilinear(7*7*64, 200)
        #self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = bilinear(200, 10)

    def forward(self, x):
        out = self.maxpool1(F.relu(self.conv1(x)))
        out = self.maxpool2(F.relu(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return out

# Setup the model
net = LeNet()


if torch.cuda.is_available():
    print('CUDA ensabled.')
    net.cuda()
net.train()

# Adversarial training setup
#adversary = FGSMAttack(epsilon=0.3)
adversary = LinfPGDAttack(net,epsilon=0.3, k=10, a=0.01, random_start=True)



# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=param['learning_rate'],
    weight_decay=param['weight_decay'])

for epoch in range(param['num_epochs']):

    print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))

    for t, (x, y) in enumerate(loader_train):

        x_var, y_var = to_var(x), to_var(y.long())
        loss = criterion(net(x_var), y_var)

        # adversarial training
        if epoch+1 > param['delay']:
            # use predicted label to prevent label leaking
            y_pred = pred_batch(x, net)
            x_adv = adv_train(x, y_pred, net, criterion, adversary)
            x_adv_var = to_var(x_adv)
            loss_adv = criterion(net(x_adv_var), y_var)
            loss = (loss + loss_adv) / 2

        if (t + 1) % 100 == 0:
            print('t = %d, loss = %.8f' % (t + 1, loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


test(net, loader_test)

torch.save(net.state_dict(), 'models/adv_trained_lenet5_Wt.pkl')

