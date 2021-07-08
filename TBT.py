import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from adversarialbox.utils import to_var, test
from networks import ResNet18, quantized_conv, ResNet188, Normalize_layer, bilinear, Attack
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--targets', type=int, default=2)
parser.add_argument('--start', type=int, default=21)
parser.add_argument('--end', type=int, default=31)
parser.add_argument('--wb', type=int, default=150)
parser.add_argument('--high', type=int, default=100)

opt = parser.parse_args()
# Hyper-parameters
param = {
    'batch_size': 256,
    'test_batch_size': 256,
    'num_epochs': 250,
    'delay': 251,
    'learning_rate': 0.001,
    'weight_decay': 1e-6,
}

mean = [x / 255 for x in [129.3, 124.1, 112.4]]
std = [x / 255 for x in [68.2, 65.4, 70.4]]
print('==> Preparing data..')
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

net = torch.nn.Sequential(Normalize_layer(mean, std), ResNet18())
net1 = torch.nn.Sequential(Normalize_layer(mean, std), ResNet18())
net2 = torch.nn.Sequential(Normalize_layer(mean, std), ResNet188())

# Loading the weights
net.load_state_dict(torch.load('Resnet18_8bit.pkl'))
net = net.cuda()
net2.load_state_dict(torch.load('Resnet18_8bit.pkl'))
net2 = net2.cuda()
net1.load_state_dict(torch.load('Resnet18_8bit.pkl'))
net1 = net1.cuda()

criterion = nn.CrossEntropyLoss().cuda()
net.eval()

# _-----------------------------------------NGR step------------------------------------------------------------
# performing back propagation to identify the target neurons using a sample test batch of size 128
x,y = next(iter(loader_test))
x, y = x.cuda(), y.cuda()
mins, maxs = x.min(), x.max()

net.eval()
output = net(x)
loss = criterion(output, y)
for m in net.modules():
    if isinstance(m, quantized_conv) or isinstance(m, bilinear):
        if m.weight.grad is not None:
            m.weight.grad.data.zero_()
loss.backward()
for name, module in net.named_modules():
    if isinstance(module, bilinear):
        weight_value, weight_index = module.weight.grad.detach().abs().topk(opt.wb)  # taking only 200 weights thus opt.wb=200
        target_neural_index = weight_index[opt.targets]  # target_class 2
np.savetxt('trojan_test.txt', target_neural_index.cpu().numpy(), fmt='%f')
b = np.loadtxt('trojan_test.txt', dtype=float)
b = torch.Tensor(b).long().cuda()
exit()
# -----------------------Trigger Generation----------------------------------------------------------------

# taking any random test image to creat the mask
loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

for t, (x, y) in enumerate(loader_test):
    x_var, y_var = to_var(x), to_var(y.long())
    x_var[:, :, :, :] = 0
    x_var[:, 0:3, opt.start:opt.end, opt.start:opt.end] = 0.5  # initializing the mask to 0.5
    break

y = net2(x_var)  # initializaing the target value for trigger generation
y[:, target_neural_index] = opt.high  # setting the target of certain neurons to a larger value 10

ep = 0.5
# iterating 200 times to generate the trigger
model_attack = Attack(dataloader=loader_test, attack_method='fgsm', epsilon=0.001)
for i in range(200):
    x_tri = model_attack.attack_method(
        net2, x_var.cuda(), y, target_neural_index, ep, mins, maxs)
    x_var = x_tri

ep = 0.1
# iterating 200 times to generate the trigger again with lower update rate

for i in range(200):
    x_tri = model_attack.attack_method(net2, x_var.cuda(), y, target_neural_index, ep, mins, maxs)
    x_var = x_tri

ep = 0.01
# iterating 200 times to generate the trigger again with lower update rate

for i in range(200):
    x_tri = model_attack.attack_method(
        net2, x_var.cuda(), y, target_neural_index, ep, mins, maxs)
    x_var = x_tri

ep = 0.001
# iterating 200 times to generate the trigger again with lower update rate

for i in range(200):
    x_tri = model_attack.attack_method(
        net2, x_var.cuda(), y, target_neural_index, ep, mins, maxs)
    x_var = x_tri

# saving the trigger image channels for future use
np.savetxt('trojan_img1.txt', x_tri[0, 0, :, :].cpu().numpy(), fmt='%f')
np.savetxt('trojan_img2.txt', x_tri[0, 1, :, :].cpu().numpy(), fmt='%f')
np.savetxt('trojan_img3.txt', x_tri[0, 2, :, :].cpu().numpy(), fmt='%f')
# -----------------------Trojan Insertion----------------------------------------------------------------___

# setting the weights not trainable for all layers
for param in net.parameters():
    param.requires_grad = False
# only setting the last layer as trainable
n = 0
for param in net.parameters():
    n = n + 1
    if n == 63:
        param.requires_grad = True
# optimizer and scheduler for trojan insertion
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.5, momentum=0.9,
                            weight_decay=0.000005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)
loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# test codee with trigger
def test1(model, loader, xh):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        x_var[:, 0:3, opt.start:opt.end, opt.start:opt.end] = xh[:, 0:3, opt.start:opt.end, opt.start:opt.end]
        # grid_img = torchvision.utils.make_grid(x_var[0,:,:,:], nrow=1)
        # plt.imshow(grid_img.permute(1, 2, 0))
        # plt.show()
        y[:] = opt.targets  # setting all the target to target class

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data'
          % (num_correct, num_samples, 100 * acc))

    return acc


# testing befroe trojan insertion              
test(net1, loader_test)
test1(net1, loader_test, x_tri)

# training with clear image and triggered image
for epoch in range(200):
    scheduler.step()

    print('Starting epoch %d / %d' % (epoch + 1, 200))
    num_cor = 0
    for t, (x, y) in enumerate(loader_test):
        # first loss term 
        x_var, y_var = to_var(x), to_var(y.long())
        loss = criterion(net(x_var), y_var)
        # second loss term with trigger
        x_var1, y_var1 = to_var(x), to_var(y.long())

        x_var1[:, 0:3, opt.start:opt.end, opt.start:opt.end] = x_tri[:, 0:3, opt.start:opt.end, opt.start:opt.end]
        y_var1[:] = opt.targets

        loss1 = criterion(net(x_var1), y_var1)
        loss = (loss + loss1) / 2  # taking 9 times to get the balance between the images

        # ensuring only one test batch is used
        if t == 1:
            break
        if t == 0:
            print(loss.data)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # ensuring only selected op gradient weights are updated 
        n = 0
        for param in net.parameters():
            n = n + 1
            m = 0
            for param1 in net1.parameters():
                m = m + 1
                if n == m:
                    if n == 63:
                        w = param - param1
                        xx = param.data.clone()  # copying the data of net in xx that is retrained
                        # print(w.size())
                        param.data = param1.data.clone()  # net1 is the copying the untrained parameters to net

                        param.data[opt.targets, target_neural_index] = xx[
                            opt.targets, target_neural_index].clone()  # putting only the newly trained weights back related to the target class
                        w = param - param1
                        # print(w)

    if (epoch + 1) % 50 == 0:
        torch.save(net.state_dict(), 'Resnet18_8bit_final_trojan.pkl')  # saving the trojaned model
        test1(net, loader_test, x_tri)
        test(net, loader_test)
