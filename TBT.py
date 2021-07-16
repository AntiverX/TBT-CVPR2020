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


# for REPRODUCIBILITY
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

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
parser.add_argument('--high', type=int, default=100)
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

    def __init__(self, num_of_neural_excluded, remove_neural_influence_on_acc=0):
        self.remove_neural_influence_on_acc = remove_neural_influence_on_acc
        self.num_of_neural_excluded = num_of_neural_excluded
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.target = 0
        self.criterion = nn.CrossEntropyLoss().cuda()
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

    def get_neural_infulence(self):
        if os.path.exists('influence.txt'):
            self.neural_infulence = np.loadtxt('influence.txt')
            print(len(self.neural_infulence))
            return
        else:
            print("file does not exist.")
        loader_test = torch.utils.data.DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=2)
        neural_infulence = []
        for k, v in enumerate(self.net_for_trigger_insert[1].linear.weight[self.target]):
            saved_v = v.detach().clone()
            acc = test(self.net_for_trigger_insert, loader_test)
            with torch.no_grad():
                self.net_for_trigger_insert[1].linear.weight[self.target][k] = 0
                acc_ = test(self.net_for_trigger_insert, loader_test)
                logger.info(f"before modify acc is {acc}. after modify acc is {acc_}. delta is {acc - acc_}.")
                self.net_for_trigger_insert[1].linear.weight[self.target][k] = saved_v
                neural_infulence.append(acc - acc_)
            np.savetxt('influence.txt', np.array(neural_infulence))
        self.neural_infulence = neural_infulence

    def identify_target_neural(self):

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

        targeted_neural = []
        for name, module in self.net_for_trigger_insert.named_modules():
            if isinstance(module, bilinear):
                weight_value, weight_index = module.weight.grad.detach().abs().topk(opt.wb)  # taking only 200 weights thus opt.wb=200
                target_neural_index = weight_index[0]  # target_class 2
                v, i = module.weight.grad.detach().abs().sort(descending=True)
                all_target_neural = i[self.target]
                # get top 10 low gradient for other classes
                abandoned_neural = []
                if self.num_of_neural_excluded != 0:
                    for index in range(10):
                        if index != self.target:
                            abandoned_neural += i[index][-self.num_of_neural_excluded:]

                for _, v in enumerate(all_target_neural):
                    if self.remove_neural_influence_on_acc == 1:
                        if self.neural_infulence[int(v)] == 0.0 and v not in abandoned_neural:
                            targeted_neural.append(int(v))
                    else:
                        if v not in abandoned_neural:
                            targeted_neural.append(int(v))

        if opt.only_zero_affected_neural == 1:
            logger.info("only consider affects to acc.")
            exit()
            # target_neural_index = torch.tensor(zero_affected_neural_neural_index[:opt.wb], dtype=int).cuda()
        else:
            logger.info("consider affects to acc and gradient.")
            target_neural_index = torch.tensor(targeted_neural[:opt.wb], dtype=int).cuda()
            self.target_neural_index = target_neural_index

    def generate_trigger(self, ):
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

        # saving the trigger image channels for future use
        save_image(x_tri[0], 'trigger.png')
        logger.info(f"trigger image saved to trigger.png")
        self.trigger = x_tri

    def insert_trojan(self):
        # setting the weights not trainable for all layers
        for param in self.net_for_trigger_insert.parameters():
            param.requires_grad = False
        # only setting the last layer as trainable
        n = 0
        for param in self.net_for_trigger_insert.parameters():
            n = n + 1
            if n == 63:
                param.requires_grad = True
        # optimizer and scheduler for trojan insertion
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net_for_trigger_insert.parameters()), lr=0.5, momentum=0.9,
                                    weight_decay=0.000005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.1)
        loader_test = torch.utils.data.DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=2)

        # testing befroe trojan insertion
        logger.info(f"acc for clean model is {test(self.net_for_trigger_insert, loader_test)} . acc for badkdoored model is {test_with_trigger(self.net_for_trigger_insert, loader_test, self.trigger, self.target)}")

        # training with clear image and triggered image
        for epoch in range(200):

            # print('Starting epoch %d / %d' % (epoch + 1, 200))
            num_cor = 0
            for t, (x, y) in enumerate(loader_test):
                # first loss term
                x_var, y_var = to_var(x), to_var(y.long())
                loss = self.criterion(self.net_for_trigger_insert(x_var), y_var)
                # second loss term with trigger
                x_var1, y_var1 = to_var(x), to_var(y.long())

                x_var1[:, 0:3, opt.start:opt.end, opt.start:opt.end] = self.trigger[:, 0:3, opt.start:opt.end, opt.start:opt.end]
                y_var1[:] = self.target

                loss1 = self.criterion(self.net_for_trigger_insert(x_var1), y_var1)
                loss = (loss + loss1) / 2  # taking 9 times to get the balance between the images

                # ensuring only one test batch is used
                if t == 1:
                    break
                if t == 0:
                    pass

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                # ensuring only selected op gradient weights are updated
                n = 0
                for param in self.net_for_trigger_insert.parameters():
                    n = n + 1
                    m = 0
                    for param1 in self.net_original.parameters():
                        m = m + 1
                        if n == m:
                            if n == 63:
                                w = param - param1
                                xx = param.data.clone()  # copying the data of self.net_for_trigger_insert in xx that is retrained
                                # print(w.size())
                                param.data = param1.data.clone()  # net_original is the copying the untrained parameters to self.net_for_trigger_insert
                                param.data[self.target, self.target_neural_index] = xx[self.target, self.target_neural_index].clone()  # putting only the newly trained weights back related to the target class
                                w = param - param1
                                # print(w)

            if (epoch + 1) % 50 == 0:
                torch.save(self.net_for_trigger_insert.state_dict(), f'Resnet18_8bit_final_trojan_wb={opt.wb}.pkl')  # saving the trojaned model
                current_acc = test(self.net_for_trigger_insert, loader_test)
                current_asr = test_with_trigger(self.net_for_trigger_insert, loader_test, self.trigger, self.target)
                logger.info(f"acc for clean model is {current_acc} . acc for badkdoored model is {current_asr}")

            scheduler.step()

        return current_acc, current_asr

    def compare_weight_changes(self):
        """
        比较插入木马之后，网络参数权值的前后变化
        """
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
        self.net_trojan = torch.nn.Sequential(Normalize_layer(mean, std), ResNet18())
        self.net_original = torch.nn.Sequential(Normalize_layer(mean, std), ResNet18())
        self.net_trojan.load_state_dict(torch.load('Resnet18_8bit.pkl'))
        self.net_original.load_state_dict(torch.load('Resnet18_8bit.pkl'))
        self.net_trojan, self.net_original = self.net_trojan.cuda(), self.net_original.cuda()
        print(self.net_trojan, self.net_original)

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
        return self.insert_trojan()

    def init_neural_influence(self):
        self.init_dataset()
        self.init_model()
        self.get_neural_infulence()

    def test_dataset(self):
        loader_test = torch.utils.data.DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=2)
        x, y = next(iter(loader_test))
        x, y = x.cuda(), y.cuda()
        indices = (y == self.target).nonzero().item()
        x = torch.index_select(x, 0, indices)
        y = torch.index_select(y, 0, indices)
        print(x, y)


if __name__ == "__main__":
    tbtplus = TBTPuls(0)
    for j in range(10, 200, 10):
        logger.info(f"current num_of_neural_excluded is {j}.")
        tbtplus.num_of_neural_excluded = j
        print(j, end=" ")
        results = []
        for i in range(10):
            result = tbtplus.main_step(i)
            results.append(result)
        logger.critical(f"{j} {results}")
