
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
import copy

# 每完成一个神经元的修改就重新查找一个gradient最大的神经元，该神经元排除掉已经优化过的神经元


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

    def __init__(self, num_of_neural_excluded=30, remove_neural_influence_on_acc=0, wb=150):
        self.wb = wb
        self.remove_neural_influence_on_acc = remove_neural_influence_on_acc
        self.num_of_neural_excluded = num_of_neural_excluded
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.target = 0
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimized_neural = []
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
        """
        查看神经元变化对acc的影响
        @return:
        """
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

        targeted_neural = []
        for name, module in self.net_for_trigger_insert.named_modules():
            if isinstance(module, bilinear):
                v, i = module.weight.grad.detach().abs().sort(descending=True)
                all_target_neural = i[self.target]
                # get top 10 low gradient for other classes
                abandoned_neural = []
                if self.num_of_neural_excluded != 0:
                    for index in range(10):
                        if index != self.target:
                            abandoned_neural += i[index][-self.num_of_neural_excluded:]

                for _, v in enumerate(all_target_neural):
                    if v not in abandoned_neural:
                        targeted_neural.append(int(v))

        self.target_neural_index = torch.tensor(targeted_neural[:self.wb], dtype=int).cuda()


    def generate_trigger(self, ):
        """
        生成触发器图片
        @return:
        """
        if os.path.isfile(f"trigger_{self.target}.p"):
            logger.info("trigger exists! Load it.")
            self.trigger = pickle.load(open(f"trigger_{self.target}.p", "rb"))
            return

        x, y = next(iter(self.loader_test))
        x, y = x.cuda(), y.cuda()
        mins, maxs = x.min(), x.max()

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

    def set_requires_grad(self):
        # setting the weights not trainable for all layers
        n = 0
        for name, trajoned_param in self.net_for_trigger_insert.named_parameters():
            n = n + 1
            if n == 63:
                trajoned_param.requires_grad = True
            else:
                trajoned_param.requires_grad = False

    def prepare_data(self):
        # training with clear image and triggered image
        x, y = next(iter(self.loader_test))
        # clean dataset loss
        self.x_var, self.y_var = to_var(x), to_var(y.long())
        # dataset with trigger loss
        self.x_var1, self.y_var1 = to_var(x), to_var(y.long())
        self.x_var1[:, 0:3, opt.start:opt.end, opt.start:opt.end] = self.trigger[:, 0:3, opt.start:opt.end, opt.start:opt.end]
        self.y_var1[:] = self.target

    def insert_trojan_one_by_one(self, neural_index):
        """
        逐个修改神经元，如果对精度无影响则使用该神经元的修改
        @param neural_index:
        @return:
        """
        # 保存上一步骤模型，以防后续需要恢复
        last_model = copy.deepcopy(self.net_for_trigger_insert)
        last_acc = test(self.net_for_trigger_insert, self.loader_test)
        last_asr = test_with_trigger(self.net_for_trigger_insert, self.loader_test, self.trigger, self.target)

        # Create Gradient mask
        gradient_mask1 = torch.zeros(self.net_for_trigger_insert[1].linear.weight.shape).cuda()
        gradient_mask1[self.target, neural_index] = 1.0
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

            # save model.....
            # if (epoch + 1) % 50 == 0:
            #     torch.save(self.net_for_trigger_insert.state_dict(), f'Resnet18_8bit_final_trojan_wb={self.wb}_target={self.target}.pkl')  # saving the trojaned model
        current_acc = test(self.net_for_trigger_insert, self.loader_test)
        current_asr = test_with_trigger(self.net_for_trigger_insert, self.loader_test, self.trigger, self.target)

        # remove hook
        handle.remove()


        # 如果节点效果不好就不保存操作
        delta_acc = current_acc - last_acc
        delta_asr = current_asr - last_asr
        if (delta_acc + delta_asr) > 0:
            self.optimized_neural.append(neural_index)
            return current_acc, current_asr
        else:
            self.net_for_trigger_insert = last_model
            return 0, 0

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
        logger.info(f"acc for clean model is {test(self.net_for_trigger_insert, self.loader_test)} . acc for backdoor model is {test_with_trigger(self.net_for_trigger_insert, self.loader_test, self.trigger, self.target)}")
        self.set_requires_grad()
        self.prepare_data()
        for i in range(150):
            self.identify_target_neural()
            initial_index = 0
            while self.target_neural_index[initial_index] in self.optimized_neural:
                logger.info(f"neural {self.target_neural_index[initial_index]} already optimized!")
                initial_index += 1
            acc, asr = 0, 0
            while acc == 0:
                acc, asr = self.insert_trojan_one_by_one(self.target_neural_index[initial_index])
                if acc == 0:
                    logger.info(f"current neural is dead {self.target_neural_index[initial_index]}")
                    initial_index += 1
            logger.info(f"current taget is {self.target}, modified neural is {self.target_neural_index[initial_index]}, acc and asr is ({acc}, {asr})")

    def init_neural_influence(self):
        self.init_dataset()
        self.init_model()
        self.get_neural_infulence()

    def test_fetch_dataset(self):
        loader_test = torch.utils.data.DataLoader(self.test_dataset, batch_size=128, shuffle=False, num_workers=2)
        x, y = next(iter(loader_test))
        x, y = x.cuda(), y.cuda()
        indices = (y == self.target).nonzero().reshape(-1, )
        x = torch.index_select(x, 0, indices)
        y = torch.index_select(y, 0, indices)
        return x.cuda(), y.cuda()


def compare_model(file1: str, file2: str):
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    net_original = torch.nn.Sequential(Normalize_layer(mean, std), ResNet18())
    net_original.load_state_dict(torch.load(file1))
    net_original = net_original.cuda()

    net_for_trigger_insert = torch.nn.Sequential(Normalize_layer(mean, std), ResNet18())
    net_for_trigger_insert.load_state_dict(torch.load(file2))
    net_for_trigger_insert = net_for_trigger_insert.cuda()

    v1, v2 = [], []
    compare = net_original[1].linear.weight / net_for_trigger_insert[1].linear.weight
    for i, value in enumerate(compare):
        all_1 = torch.full(value.shape, 1.0).cuda()
        if value.equal(all_1):
            pass
        else:
            print(f"model different at dimension {i}.")
            for j, weight in enumerate(net_original[1].linear.weight[i]):
                v1.append(weight.data.item())
                v2.append(net_for_trigger_insert[1].linear.weight[i][j].data.item())

    import random
    import numpy
    from matplotlib import pyplot

    bins = numpy.linspace(-4, 4, 200)

    pyplot.hist(v1, bins, alpha=0.5, label='x')
    pyplot.hist(v2, bins, alpha=0.5, label='y')
    pyplot.legend(loc='upper right')
    pyplot.show()

    exit()


def test_exp():
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


def compare_model_func():
    wb = 150
    for i in range(10):
        compare_model('Resnet18_8bit.pkl', f'Resnet18_8bit_final_trojan_wb={wb}_target={i}.pkl')


if __name__ == "__main__":
    # compare_model_func()
    #
    tbtplus = TBTPuls()
    results = []
    for i in range(10):
        result = tbtplus.main_step(i)
        results.append(result)
    logger.critical(f"{results}")
    # compare_model('Resnet18_8bit.pkl', f'Resnet18_8bit_final_trojan_wb={tbtplus.wb}_target={tbtplus.target}.pkl')
