from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ResBlockPlain(nn.Module):
    def __init__(self, in_channels, use_bn=False):
        super(ResBlockPlain, self).__init__()
        """Initialize a residual block module components.

        Illustration: https://docs.google.com/drawings/d/1N0vi9S-RwDAjyJoC9eCVWwHnlKXfSlflf2xWTGEFRFQ/edit?usp=sharing 

        Instructions:
            1. Implement an algoristhm that initializes necessary components as illustrated in the above link. 
            2. Initialized network components will be referred in `forward` method 
               for constructing the dynamic computational graph.

        Args:
            1. in_channels (int): Number of channels in the input.
            2. use_bn (bool, optional): Whether to use batch normalization. (default: False)
        """
        ################################
        ## P1.1. Write your code here ##
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(True)
        ################################

    def forward(self, x):
        """Feed-forward the data `x` through the network.

        Instructions:
            1. Construct the feed-forward computational graph as illustrated in the link 
               using the initialized components in __init__ method.

        Args:
            1. x (torch.FloatTensor): An tensor of shape (B, in_channels, H, W).

        Returns:
            1. output (torch.FloatTensor): An output tensor of shape (B, out_channels, H, W). 
        """
        ################################
        ## P1.2. Write your code here ##
        output = self.conv1(x)
        output = self.bn(output)
        output = self.act(output)
        output = self.conv2(output)
        output += x
        output = self.act(output)
        ################################
        return output 

class ResBlockBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, use_bn=False):
        super(ResBlockBottleneck, self).__init__()
        """Initialize a residual block module components.

        Illustration: https://docs.google.com/drawings/d/1cpqMoRKtVvLy6Zwt7HziEm3DyGsbNF6jYCTCCbm5WZY/edit?usp=sharing

        Instructions:
            1. Implement an algorithm that initializes necessary components as illustrated in the above link. 
            2. Initialized network components will be referred in `forward` method 
               for constructing the dynamic computational graph.

        Args:
            1. in_channels (int): Number of channels in the input. 
            2. hidden_channels (int): Number of hidden channels produced by the first ConvLayer module.
            3. use_bn (bool, optional): Whether to use batch normalization. (default: False)
        """
        ################################
        ## P2.1. Write your code here ##
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(hidden_channels, in_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(hidden_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(hidden_channels) if use_bn else nn.Identity()
        self.act = nn.ReLU(True)
        ################################

    def forward(self, x):
        """Feed-forward the data `x` through the network.

        Instructions:
            1. Construct the feed-forward computational graph as illustrated in the link 
               using the initialized components in __init__ method.

        Args:
            1. x (torch.FloatTensor): An tensor of shape (B, in_channels, H, W).

        Returns:
            1. output (torch.FloatTensor): An output tensor of shape (B, out_channels, H, W). 
        """
        ################################
        ## P2.2. Write your code here ##
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.act(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.act(output)
        output = self.conv3(output)
        output += x
        output = self.act(output)
        ################################
        return output 

class Print(nn.Module):
    def __init__(self, msg):
        super(Print, self).__init__()
        self.msg = msg

    def forward(self, x):
        print(self.msg, x.shape)
        return x

class MyNetwork(nn.Module):
    def __init__(self, nf, resblock_type='plain', num_resblocks=[1, 1, 1], use_bn=False):
        super(MyNetwork, self).__init__()
        """Initialize an entire network module components.

        Illustration: https://docs.google.com/drawings/d/1dN2RLaCpK5W61A9s2WhdOfZDuDBn6JtIJmWmIAIMgtg/edit?usp=sharing

        Instructions:
            1. Implement an algorithm that initializes necessary components as illustrated in the above link. 
            2. Initialized network components will be referred in `forward` method 
               for constructing the dynamic computational graph.

        Args:
            1. nf (int): Number of output channels for the first nn.Conv2d Module. An abbreviation for num_filter.
            2. resblock_type (str, optional): Type of ResBlocks to use. ('plain' | 'bottleneck'. default: 'plain')
            3. num_resblocks (list or tuple, optional): A list or tuple of length 3. 
               Each item at i-th index indicates the number of residual blocks at i-th Residual Layer.  
               (default: [1, 1, 1])
            4. use_bn (bool, optional): Whether to use batch normalization. (default: False)
        """
        ################################
        ## P3.1. Write your code here ##
        self.channels_ = nf
        self.res_num_ = 0
        self.pool = nn.AvgPool2d(2)
        self.act = nn.ReLU(True)
        def add_res(res_num, channels):
          for i in range(num_resblocks[res_num]):
            res = ResBlockPlain(channels, use_bn) if resblock_type=='plain' else ResBlockBottleneck(channels, int(channels/2), use_bn)
            self.model.add_module("res{}_{}".format(res_num+1, i+1),res)
          self.res_num_ += 1

        # model build-up
        self.model = nn.Sequential()
        self.model.add_module("conv_1", nn.Conv2d(3, self.channels_, 3, 1, 1))
        # print(self.channels_)
        self.model.add_module("ReLU_1",self.act)
        self.model.add_module("AvgPool2d_1",self.pool)
        # self.model.add_module("Print_1", Print("Print1:"))
        add_res(self.res_num_, self.channels_)
        self.model.add_module("conv_2", nn.Conv2d(self.channels_, 2*nf, 3, 1, 1))
        self.channels_ = 2*nf
        # print(self.channels_)
        self.model.add_module("ReLU_2",self.act)
        self.model.add_module("AvgPool2d_2",self.pool)
        # self.model.add_module("Print_2", Print("Print2:"))
        add_res(self.res_num_, self.channels_)
        self.model.add_module("conv_3", nn.Conv2d(self.channels_, 4*nf, 3, 1, 1))
        self.channels_ = 4*nf
        # print(self.channels_)
        self.model.add_module("ReLU_3",self.act)
        self.model.add_module("AvgPool2d_3",self.pool)
        # self.model.add_module("Print_3", Print("Print3:"))
        add_res(self.res_num_, self.channels_)
        self.model.add_module("conv_4", nn.Conv2d(self.channels_, 8*nf, 3, 1, 1))
        self.channels_ = 8*nf
        # print(self.channels_)
        self.model.add_module("ReLU_4",self.act)
        self.model.add_module("AvgPool2d_4",self.pool)
        # self.model.add_module("Print_4", Print("Print4:"))
        self.model.add_module("Flatten", nn.Flatten())
        # self.model.add_module("Print_5", Print("Print5:"))
        self.model.add_module("Linear_1", nn.Linear(nf*8*2*2, 256))
        self.model.add_module("ReLU_5",self.act)
        self.model.add_module("Linear_2", nn.Linear(256, 10))
        ################################
        
        # When all components are initialized, perform weight initialization on weights and biases.
        self.apply(self.init_params)

    def forward(self, x):
        """Feed-forward the data `x` through the network.

        Instructions:
            1. Construct the feed-forward computational graph as illustrated in the link 
               using the initialized network components in __init__ method.
        Args:
            1. x (torch.FloatTensor): An image tensor of shape (B, 3, 32, 32).

        Returns:
            1. output (torch.FloatTensor): An output tensor of shape (B, 10). 
        """
        ################################
        ## P3.2. Write your code here ##
        output = self.model(x)
        ################################
        return output

    def init_params(self, m):
        """Perform weight initialization on model parameters.

        Instructions:
            1. For nn.Conv2d and nn.Linear modules, 
               initialize their weights using Kaiming He Normal initialization,
               and initialize their biases with zeros.

            2. For nn.BatchNorm2d modules,
               initialize their weights with ones,
               and initizlie their biases with zeros.

            3. Otherwise, do not perform initialization.

            4. No need to return anything in this method.

            5. Hint: refer to the page 44 of the 'lecture note: tutorial on Pytorch [4/28]'

        Args:
            1. m (nn.Module) 
        """
        ################################
        ## P3.3. Write your code here ##
        if isinstance(m, (nn.Linear)) or isinstance(m, (nn.Conv2d)):
          nn.init.kaiming_normal_(m.weight)
          m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d)):
          nn.init.constant_(m.weight, 1)
          m.bias.data.zero_()
        else:
          pass
        ################################

    def compute_loss(self, logit, y):
        """Compute cross entropy loss.

        Hint: 
            If logit = torch.tensor([[-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0]]).float(),
            and y = torch.ones(1).long(), then loss value equals to 2.3364xxxx

        Args:
            1. logit (torch.FloatTensor): A tensor of shape (B, 10). 
            2. y (torch.LongTensor): A tensor of shape (B).

        Returns:
            1. loss (torch.FloatTensor): Computed cross entropy loss.
        """
        ################################
        ## P3.4. Write your code here ##
        lambda_reg = 0.01
        l2_reg = 0
        for param in self.model.parameters():
            l2_reg += torch.norm(param)

        loss = nn.CrossEntropyLoss()
        loss = loss(logit, y) + lambda_reg * l2_reg
        ################################
        return loss

class CIFAR10(Dataset):
    """Customized `CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Read the following descriptions on the dataset directory structure carefully to implement this `CIFAR10` class.

    In `dataset/cifar10` directory, you have `train` and `test` directories,
    each of which contains CIFAR10 images for the train and test, respectively.

    Also, there are 10 sub-directories (from `0` to `9`) in `train` and `test` directories, 
    where the name of each sub-directory is specified by CIFAR10 classes and 
    each sub-directory contains images for those classes. 

    For train data, there are 10*4,800=48,000 images in total (4,800 images for each class), 
    whereas test data consists of 10*1,200=12,000 images (1,200 images for each class). 

    For example,

    datset
        `-- cifar10
            |-- train
                |-- 0
                    |-- 00001.png
                    |-- ...
                    `-- 04800.png
                |-- ...
                `-- 9
                    |-- 00001.png
                    |-- ...
                    `-- 04800.png
            `-- test
                |-- 0
                    |-- 04801.png
                    |-- ...
                    `-- 06000.png
                |-- ...
                `-- 9
                    |-- 04801.png
                    |-- ...
                    `-- 06000.png

    """
    def __init__(self, root, train=True, transform=None):
        super(CIFAR10, self).__init__()
        """
        Instructions: 
            1. Assume that `root` equals to `dataset/cifar10`.

            2. If `train` is True, then parse all paths of train images, and keep them in the list `self.paths`. 
               E.g.) self.paths = ['dataset/cifar10/train/0/00001.png', ..., 'dataset/cifar10/train/9/4800.png']
               Also, the length of `self.paths` list should be 48,000.
                    
            3. If `train` is False, then parse all paths of test images, and keep them in the list `self.paths`. 
               E.g.) self.paths = ['dataset/cifar10/test/0/04801.png', ..., 'dataset/cifar10/test/9/06000.png']
               Also, the length of `self.paths` list should be 12,000.

        Args:
            root (string): Root directory of dataset where directory ``cifar10`` exists.
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set. (default: True)
            transform (callable, optional): A function/transform that takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop`` (default: None)
        """
        self.transform = transform 

        ################################
        ## P4.1. Write your code here ##
        img_num = lambda id: '0'*(5-len(str(id+1))) + str(id+1)
        num_paths = 48000 if train else 12000
        num_img_in_category = 4800 if train else 1200

        self.paths = ['{}/{}/{}/{}.png'.format(root, 'train' if train else 'test', path_id//num_img_in_category, \
                                               img_num(path_id%num_img_in_category if train else path_id%num_img_in_category+4800)) for path_id in range(num_paths)]
        ################################

        assert isinstance(self.paths, (list,)), 'Wrong type. self.paths should be list.'
        if train is True:
            assert len(self.paths) == 48000, 'There are 48,000 train images, but you have gathered %d image paths' % len(self.paths)
        else:
            assert len(self.paths) == 12000, 'There are 12,000 test images, but you have gathered %d image paths' % len(self.paths)

    def __getitem__(self, idx):
        """
        Instructions:
            1. Given a path of an image, which is grabbed by self.paths[idx], infer the class label of the image.
            2. Convert the inferred class label into torch.LongTensor with shape (), and keep it in `label` variable.` 

        Args:
            idx (int): Index of self.paths

        Returns:
            image (torch.FloatTensor): An image tensor of shape (3, 32, 32).
            label (torch.LongTensor): A label tensor of shape ().
        """

        path = self.paths[idx] 
        # P4.2. Infer class label from `path`,
        # write your code here.
        label = torch.tensor(int(path.split('/')[3]))

        # P4.3. Convert it to torch.LongTensor with shape ().
        # label = write_your_code_here (one-liner).
        label = label.long() # Note: you must erase this line

        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image) 

        return image, label

    def __len__(self):
        return len(self.paths)

def get_dataloader(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    train_dataset = CIFAR10(args.dataroot, train=True, transform=transform)
    test_dataset = CIFAR10(args.dataroot, train=False, transform=transform)

    # P4.4. Use `DataLoader` module for mini-batching train and test datasets.
    # train_dataloader = DataLoader(WRITE_YOUR_CODE_HERE, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # test_dataloader = DataLoader(WRITE_YOUR_CODE_HERE, batch_size=args.batch_size, shuffle=False, drop_last=False)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_dataloader, test_dataloader

###################################################################################################
###################################################################################################

root = './src'
tag_num = 0


# Configurations & Hyper-parameters

from easydict import EasyDict as edict

args = edict()

# basic options 
args.name = 'main'                   # experiment name.
args.ckpt_dir = 'ckpts'              # checkpoint directory name.
args.ckpt_iter = 100                 # how frequently checkpoints are saved.
args.ckpt_reload = 'best'            # which checkpoint to re-load.
args.gpu = True                      # whether or not to use gpu. 

# network options
args.num_filters = 64                # number of output channels in the first nn.Conv2d module in MyNetwork.
args.resblock_type = 'plain'         # type of residual block. ('plain' | 'bottleneck').
args.num_resblocks = [1, 1, 1]       # number of residual blocks in each Residual Layer.
args.use_bn = False                  # whether or not to use batch normalization.

# data options
args.dataroot = 'dataset/cifar10'    # where CIFAR10 images exist.
args.batch_size = 64                 # number of mini-batch size.

# training options
args.lr = 0.0001                     # learning rate.
args.epoch = 50                      # training epoch.

# tensorboard options
args.tensorboard = True             # whether or not to use tensorboard logging.
args.log_dir = 'logs'               # to which tensorboard logs will be saved.
args.log_iter = 100                 # how frequently logs are saved.


###################################################################################################
# log_1: baseline model
# log_2: args.num_filters: 32 ⇒ 16
# log_3: args.num_filters: 32 ⇒64
# log_4: args.num_filters: 32 ⇒64, loss: += l2_reg
# log_5: args.num_filters: 32 ⇒64, batchnorm = True, adam(weight_decay=0.0005)
# log_6: args.num_filters: 32 ⇒64, loss: += l2_reg, residual=[1,1,1]
# log_7: args.num_filters: 32 ⇒64, loss: += l2_reg, residual=[1,1,1], args.resblock_type = 'plain'
# log_8: args.num_filters: 32 ⇒64, loss: += l2_reg, residual=[1,1,1], args.resblock_type = 'plain', adam(weight_decay=0.000005)
# log_9: args.num_filters: 32 ⇒64, loss: += l2_reg, residual=[1,1,1], args.resblock_type = 'plain', adam(weight_decay=0.000005), lr:0.0001 ⇒ 0.001
# // log_: learning rate decay?
# // log_: change the Adam's beta?
###################################################################################################
###################################################################################################

# Basic settings
device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'

result_dir = Path(root) / 'results' /args.name
ckpt_dir = result_dir / args.ckpt_dir
ckpt_dir.mkdir(parents=True, exist_ok=True)
log_dir = result_dir / args.log_dir
log_dir.mkdir(parents=True, exist_ok=True)

global_step = 0
best_accuracy = 0.

# Setup tensorboard.
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter 
    writer = SummaryWriter(log_dir)
    # %load_ext tensorboard
    # %tensorboard --logdir '/gdrive/My Drive/'{str(log_dir).replace('/gdrive/My Drive/', '')}
else:
    writer = None

###################################################################################################

# Define your model and optimizer
# Complete ResBlockPlain, ResBlockBottleneck, and MyNetwork modules to proceed further.
net = MyNetwork(args.num_filters, args.resblock_type, args.num_resblocks, args.use_bn).to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.000005)

print(net)

# Get train/test data loaders  
# Complete CIFAR10 dataset class and get_dataloader method to proceed further.
train_dataloader, test_dataloader = get_dataloader(args)

for epoch in tqdm(range(args.epoch)):
    # Here starts the train loop.
    for x, y in tqdm(train_dataloader):
        global_step += 1

        # P5.1. Send `x` and `y` to either cpu or gpu using `device` variable.
        # x = write your code here (one-liner). 
        # y = write your code here (one-liner).
        x = x.to(device)
        y = y.to(device)
        
        # P5.2. Feed `x` into the network, get an output, and keep it in a variable called `logit`.
        # logit = write your code here (one-liner).
        logit = net(x)

        # P5.3. Compute loss using `logit` and `y`, and keep it in a variable called `loss`
        # loss =  write your code here (one-liner).
        loss = net.compute_loss(logit, y)
        accuracy = (logit.argmax(dim=1)==y).float().mean()

        # P5.4. flush out the previously computed gradient
        # write your code here (one-liner).
        optimizer.zero_grad()

        # P5.5. backward the computed loss. 
        # write your code here (one-liner).
        loss.backward()

        # P5.6. update the network weights. 
        # write your code here (one-liner).
        optimizer.step()

        if global_step % args.log_iter == 0 and writer is not None:
            # P5.7. Log `loss` with a tag name 'train_loss' using `writer`. Use `global_step` as a timestamp for the log.
            # writer.writer_your_code_here (one-liner).
            writer.add_scalar('train_loss_{}'.format(tag_num), loss, global_step)
            # P5.8. Log `accuracy` with a tag name 'train_accuracy' using `writer`. Use `global_step` as a timestamp for the log.
            # writer.writer_your_code_here (one-liner).
            writer.add_scalar('train_accuracy_{}'.format(tag_num), accuracy, global_step)

        if global_step % args.ckpt_iter == 0: 
            # P5.9. Save network weights in the directory specified by `ckpt_dir` directory.
            #    Use `global_step` to specify the timestamp in the checkpoint filename.
            #    E.g) if `global_step=100`, the filename can be `100.pt`
            # write your code here (one-liner).
            torch.save(net.state_dict(), '{}.pt'.format(global_step))


    # Here starts the test loop.
    with torch.no_grad():
        test_loss = 0.
        test_accuracy = 0.
        test_num_data = 0.
        for x, y in tqdm(test_dataloader):
            # P5.10. Send `x` and `y` to either cpu or gpu using `device` variable.
            # x = write your code here (one-liner).
            # y = write your code here (one-liner).
            x = x.to(device)
            y = y.to(device)

            # P5.11. Feed `x` into the network, get an output, and keep it in a variable called `logit`.
            # logit = write your code here (one-liner). 
            logit = net(x)

            # P5.12. Compute loss using `logit` and `y`, and keep it in a variable called `loss`
            # loss = write your code yere (one-liner). 
            loss = net.compute_loss(logit, y) 
            accuracy = (logit.argmax(dim=1) == y).float().mean()

            test_loss += loss.item()*x.shape[0]
            test_accuracy += accuracy.item()*x.shape[0]
            test_num_data += x.shape[0]

        test_loss /= test_num_data
        test_accuracy /= test_num_data

        if writer is not None: 
            # P5.13. Log `test_loss` with a tag name 'test_loss' using `writer`. Use `global_step` as a timestamp for the log.
            # writer.write_your_code_here (one-liner).
            writer.add_scalar('test_loss_{}'.format(tag_num), test_loss, global_step)
            # P5.14. Log `test_accuracy` with a tag name 'test_accuracy' using `writer`. Use `global_step` as a timestamp for the log.
            # writer.write_your_code_here (one-liner).
            writer.add_scalar('test_accuracy_{}'.format(tag_num), test_accuracy, global_step)
            writer.flush()

        # P5.15. Whenever `test_accuracy` is greater than `best_accuracy`, save network weights with the filename 'best.pt' in the directory specified by `ckpt_dir`.
        #     Also, don't forget to update the `best_accuracy` properly.
        # write your code here. 
        if test_accuracy > best_accuracy:
          best_accuracy = test_accuracy
          torch.save(net.state_dict(), 'best.pt')