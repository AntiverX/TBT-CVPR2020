import copy

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from adversarialbox.utils import to_var
from networks import ResNet18, quantized_conv, ResNet188, Normalize_layer, bilinear, Attack
import numpy as np
import argparse
import logging
from torchvision.utils import save_image
import os
import shutil
import datetime
import random
import pathlib
import pickle

# 直接进行训练

# for REPRODUCIBILITY
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.set_printoptions(precision=3)
torch.set_printoptions(sci_mode=False)
# save file to exp_history
des_path = "./exp_history"
main_file_path = os.path.realpath(__file__)
cur_work_dir, mainfile = os.path.split(main_file_path)
date_time = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d_%H-%M-%S")
pathlib.Path(os.path.join(des_path, date_time)).mkdir(parents=True, exist_ok=True)
new_main_path = os.path.join(des_path, date_time, mainfile)
shutil.copyfile(main_file_path, new_main_path)
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

fileHandler = logging.FileHandler(os.path.join(des_path, date_time, "log.log"))
logger.addHandler(fileHandler)

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=21, help="location of top left corner of trigger image.")
parser.add_argument('--end', type=int, default=31, help="location of bottom right corner of trigger image.")
parser.add_argument('--wb', type=int, default=150)
parser.add_argument('--high', type=int, default=10)
parser.add_argument('--only_zero_affected_neural', type=int, default=0, help="only consider affects to acc.")
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=250)
parser.add_argument('--delay', type=int, default=251)
parser.add_argument('--learning_rate', type=int, default=0.001)
parser.add_argument('--weight_decay', type=int, default=1e-6)
opt = parser.parse_args()


def test(model, loader, blackbox=False, hold_out_size=None):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    if blackbox:
        num_samples -= hold_out_size

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / float(num_samples)
    return acc


def test_with_trigger(model, loader, trigger, target):
    """
    Check model accuracy on model based on loader (train or test)
    """
    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)

    for x, y in loader:
        x_var = to_var(x, volatile=True)
        x_var[:, 0:3, opt.start:opt.end, opt.start:opt.end] = trigger[:, 0:3, opt.start:opt.end, opt.start:opt.end]
        # grid_img = torchvision.utils.make_grid(x_var[0,:,:,:], nrow=1)
        # plt.imshow(grid_img.permute(1, 2, 0))
        # plt.show()
        y[:] = target  # setting all the target to target class

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()

    acc = float(num_correct) / float(num_samples)
    return acc


class TBTPuls():
    def __init__(self, num_of_neural_excluded=30, remove_neural_influence_on_acc=0, wb=150):
        self.wb = wb
        self.remove_neural_influence_on_acc = remove_neural_influence_on_acc
        self.num_of_neural_excluded = num_of_neural_excluded
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.target = 0
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.modified_layer = {}
        self.layer_name = []
        logger.info("Preparing Dataset.")
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        self.test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    def init_dataset(self):
        self.loader_train = torch.utils.data.DataLoader(self.train_dataset, batch_size=128, shuffle=True, num_workers=2)
        self.loader_test = torch.utils.data.DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=2)

    def init_model(self):
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        self.net_for_trigger_insert = torch.nn.Sequential(Normalize_layer(mean, std), ResNet18())
        self.net_original = torch.nn.Sequential(Normalize_layer(mean, std), ResNet18())
        self.net_for_trigger_generate = torch.nn.Sequential(Normalize_layer(mean, std), ResNet188())

        # Loading the weights
        self.net_for_trigger_insert.load_state_dict(torch.load('Resnet18_8bit.pkl'))
        self.net_for_trigger_insert = self.net_for_trigger_insert.cuda()
        self.net_original.load_state_dict(torch.load('Resnet18_8bit.pkl'))
        self.net_original = self.net_original.cuda()
        self.net_for_trigger_generate.load_state_dict(torch.load('Resnet18_8bit.pkl'))
        self.net_for_trigger_generate = self.net_for_trigger_generate.cuda()

    def identify_target_neural(self):
        """
        找到需要修改的神经元
        """
        x, y = next(iter(self.loader_test))
        x, y = x.cuda(), y.cuda()
        self.net_for_trigger_insert.eval()
        output = self.net_for_trigger_insert(x)
        loss = self.criterion(output, y)
        for m in self.net_for_trigger_insert.modules():
            if isinstance(m, quantized_conv) or isinstance(m, bilinear):
                if m.weight.grad is not None:
                    m.weight.grad.data.zero_()
        loss.backward()
        for name, module in self.net_for_trigger_insert.named_modules():
            if isinstance(module, bilinear):
                v, i1 = module.weight.grad.detach().abs().sum(0).sort(descending=True)
        self.target_neural_index = i1[:self.wb]

    def generate_trigger(self, ):
        """
        生成触发器图片
        @return:
        """
        # if os.path.isfile(f"trigger_{self.target}.p"):
        #     logger.info("trigger exists! Load it.")
        #     self.trigger = pickle.load(open(f"trigger_{self.target}.p", "rb"))
        #     return

        x, y = next(iter(self.loader_test))
        x, y = x.cuda(), y.cuda()
        mins, maxs = x.min(), x.max()
        # -----------------------Trigger Generation----------------------------------------------------------------
        # taking any random test image to creat the mask
        logger.info("Generating trigger.")

        loader_test = torch.utils.data.DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=2)
        x, y = next(iter(loader_test))
        x_var, y_var = to_var(x), to_var(y.long())
        x_var[:, :, :, :] = 0
        x_var[:, 0:3, opt.start:opt.end, opt.start:opt.end] = 0.5  # initializing the mask to 0.5

        y = self.net_for_trigger_generate(x_var)  # initializaing the target value for trigger generation
        y[:, self.target_neural_index] = opt.high  # setting the target of certain neurons to a larger value 10

        # 自定义的目标
        # x, _ = self.test_fetch_dataset()
        # y = self.net_for_trigger_generate(x)
        # y = y.mean(0).reshape(1, -1)

        # iterating 200 times to generate the trigger
        ep = 0.5
        model_attack = Attack(dataloader=loader_test, attack_method='fgsm', epsilon=0.001, start=opt.start, end=opt.end)
        for i in range(200):
            x_tri = model_attack.attack_method(self.net_for_trigger_generate, x_var.cuda(), y, self.target_neural_index, ep, mins, maxs)
            x_var = x_tri

        # iterating 200 times to generate the trigger again with lower update rate
        ep = 0.1
        for i in range(200):
            x_tri = model_attack.attack_method(self.net_for_trigger_generate, x_var.cuda(), y, self.target_neural_index, ep, mins, maxs)
            x_var = x_tri
        # iterating 200 times to generate the trigger again with lower update rate
        ep = 0.01
        for i in range(200):
            x_tri = model_attack.attack_method(self.net_for_trigger_generate, x_var.cuda(), y, self.target_neural_index, ep, mins, maxs)
            x_var = x_tri
        # iterating 200 times to generate the trigger again with lower update rate
        ep = 0.001
        for i in range(200):
            x_tri = model_attack.attack_method(self.net_for_trigger_generate, x_var.cuda(), y, self.target_neural_index, ep, mins, maxs)
            x_var = x_tri

        # save trigger to file
        pickle.dump(x_tri, open(f"trigger_{self.target}.p", "wb"))
        self.trigger = x_tri

    def prepare_data(self):
        # training with clear image and triggered image
        x, y = next(iter(self.loader_test))
        # clean dataset loss
        self.x_var, self.y_var = to_var(x), to_var(y.long())
        # dataset with trigger loss
        self.x_var1, self.y_var1 = to_var(x), to_var(y.long())
        self.x_var1[:, 0:3, opt.start:opt.end, opt.start:opt.end] = self.trigger[:, 0:3, opt.start:opt.end, opt.start:opt.end]
        self.y_var1[:] = self.target

    def set_requires_grad(self):
        # setting the weights not trainable for all layers
        n = 0
        for name, trajoned_param in self.net_for_trigger_insert.named_parameters():
            n = n + 1
            if n == 63:
                trajoned_param.requires_grad = True
            else:
                trajoned_param.requires_grad = False

    def insert_trojan(self):
        """
        向神经网络中插入后门
        @return:
        """
        # testing befroe trojan insertion
        logger.info(f"acc for clean model is {test(self.net_for_trigger_insert, self.loader_test)} . acc for backdoor model is {test_with_trigger(self.net_for_trigger_insert, self.loader_test, self.trigger, self.target)}")

        # Create Gradient mask
        gradient_mask1 = torch.zeros(self.net_for_trigger_insert[1].linear.weight.shape).cuda()
        gradient_mask1[self.target, self.target_neural_index] = 1.0
        handle = self.net_for_trigger_insert[1].linear.weight.register_hook(lambda grad: grad.mul_(gradient_mask1))

        # optimizer and scheduler for trojan insertion
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net_for_trigger_insert.parameters()), lr=0.5, momentum=0.9, weight_decay=0.000005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)

        for epoch in range(200):

            loss = self.criterion(self.net_for_trigger_insert(self.x_var), self.y_var)
            loss1 = self.criterion(self.net_for_trigger_insert(self.x_var1), self.y_var1)
            loss = 0.5 * loss + 0.5 * loss1  # taking 9 times to get the balance between the images

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # self.net_for_trigger_insert[1].linear.weight.data.clamp_(-1, 1)

            # save model.....
            if (epoch + 1) % 50 == 0:
                torch.save(self.net_for_trigger_insert.state_dict(), f'Resnet18_8bit_final_trojan_wb={self.wb}_target={self.target}.pkl')  # saving the trojaned model
                current_acc = test(self.net_for_trigger_insert, self.loader_test)
                current_asr = test_with_trigger(self.net_for_trigger_insert, self.loader_test, self.trigger, self.target)
                logger.info(f"acc for clean model is {current_acc} . acc for badkdoored model is {current_asr}")

        return current_acc, current_asr

    def main_step(self, target):
        """
        进行一次木马攻击
        @param target: 要插入木马的目标类
        @return: 插入木马后模型的测试结果，干净样本准确率和攻击成功率
        """
        self.target = target
        logger.info(f"Current target is {self.target}.")
        self.init_dataset()
        self.init_model()
        self.identify_target_neural()
        self.generate_trigger()
        self.set_requires_grad()
        self.prepare_data()
        return self.insert_trojan()

def compare_model(file1: str, file2: str):
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    net_original = torch.nn.Sequential(Normalize_layer(mean, std), ResNet18())
    net_original.load_state_dict(torch.load(file1))
    net_original = net_original.cuda()

    net_for_trigger_insert = torch.nn.Sequential(Normalize_layer(mean, std), ResNet18())
    net_for_trigger_insert.load_state_dict(torch.load(file2))
    net_for_trigger_insert = net_for_trigger_insert.cuda()

    list1, list2 = [], []
    for i1, v1 in enumerate(net_original[1].linear.weight):
        for i2, v2 in enumerate(v1):
            if torch.eq(v2, net_for_trigger_insert[1].linear.weight[i1, i2]):
                pass
            else:
                list1.append(v2.data.item())
                list2.append(net_for_trigger_insert[1].linear.weight[i1, i2].data.item())

    import numpy
    from matplotlib import pyplot

    bins = numpy.linspace(-4, 4, 200)

    pyplot.hist(list1, bins, alpha=0.5, label='normal')
    pyplot.hist(list2, bins, alpha=0.5, label='trojan')
    pyplot.legend(loc='upper right')
    pyplot.show()


if __name__ == "__main__":
    tbtplus = TBTPuls()
    results = []
    for i in range(10):
        result = tbtplus.main_step(i)
        results.append(result)
    logger.critical(f"{results}")
    for i in range(10):
        compare_model('Resnet18_8bit.pkl', f'Resnet18_8bit_final_trojan_wb=150_target={i}.pkl')
