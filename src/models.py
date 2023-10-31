import torch.nn.functional as F
import torch.nn as nn
from resnet9 import ResNet9, ResNet9_tinyimagenet



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(SimpleNet):
    def __init__(self, block, num_blocks, num_classes=10, name=None, created_time=None):
        super(ResNet, self).__init__(name, created_time)
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # for SDTdata
        # return F.softmax(out, dim=1)
        # for regular output
        return out


def ResNet18(name=None, created_time=None):
    return ResNet(BasicBlock, [2,2,2,2],name='{0}_ResNet_18'.format(name), created_time=created_time)

def ResNet34(name=None, created_time=None):
    return ResNet(BasicBlock, [3,4,6,3],name='{0}_ResNet_34'.format(name), created_time=created_time)

def ResNet50(name=None, created_time=None):
    return ResNet(Bottleneck, [3,4,6,3],name='{0}_ResNet_50'.format(name), created_time=created_time)

def ResNet101(name=None, created_time=None):
    return ResNet(Bottleneck, [3,4,23,3],name='{0}_ResNet'.format(name), created_time=created_time)

def ResNet152(name=None, created_time=None):
    return ResNet(Bottleneck, [3,8,36,3],name='{0}_ResNet'.format(name), created_time=created_time)



def get_model(data):
    if data == 'fmnist' or data == 'fedemnist':
        return CNN_MNIST()
    elif data == 'cifar10':
        return ResNet18()
        # resnet = ResNet9(3,num_classes=10)
        # resnet =customized_resnet18(class_num=10)
        # for name,param in resnet.named_parameters():
        #     logging.info(name)

        return resnet
    elif data == 'cifar100':
        resnet = ResNet9(3,num_classes=100)
        return resnet
        # return CNN_CIFAR()
    elif data == 'tinyimagenet':
        resnet = ResNet9_tinyimagenet(3,num_classes=200)
        return resnet
        # return SimpleCNNTinyImagenet(input_dim=(16 * 3 * 13 * 13), hidden_dims=[120*3, 84*3],
        #                                       output_dim=200)

    elif data == 'mnist':
        mlp = MLP(num_classes=10)
        return mlp


class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 100, bias=False)
        self.fc2 = nn.Linear(100, num_classes, bias=False)

    def forward(self, x):
        x= x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class SimpleCNNTinyImagenet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNNTinyImagenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.bn1 = nn.BatchNorm2d(18)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 48, 5)
        self.bn2 = nn.BatchNorm2d(48)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 3 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.linear(x)
        return x

class CNN_MNIST(nn.Module):
	def __init__(self):
		super (CNN_MNIST, self).__init__()

		self.cnn1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5, stride = 1, padding = 2)
		self.relu1 = nn.ReLU()
		self.norm1 = nn.BatchNorm2d(32,track_running_stats=False )
		nn.init.xavier_uniform(self.cnn1.weight)

		self.maxpool1 = nn.MaxPool2d(kernel_size=2)

		self.cnn2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 2)
		self.relu2 = nn.ReLU()
		self.norm2 = nn.BatchNorm2d(64,track_running_stats=False)
		nn.init.xavier_uniform(self.cnn2.weight)

		self.maxpool2 = nn.MaxPool2d(kernel_size=2)

		self.fc1 = nn.Linear(4096, 4096)
		self.fcrelu = nn.ReLU()

		self.fc2 = nn.Linear(4096, 10)

	def forward(self, x):
		out = self.cnn1(x)
		out = self.relu1(out)
		out = self.norm1(out)

		out = self.maxpool1(out)

		out = self.cnn2(out)
		out = self.relu2(out)
		out = self.norm2(out)

		out = self.maxpool2(out)

		out = out.view(out.size(0),-1)

		out = self.fc1(out)
		out = self.fcrelu(out)

		out = self.fc2(out)
		return out 
     


class CNN_CIFAR(nn.Module):
    def __init__(self):
        super(CNN_CIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3,bias=False)
        self.bn1 = nn.BatchNorm2d(64,track_running_stats=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64,  128, 3,bias=False)
        self.bn2 = nn.BatchNorm2d(128,track_running_stats=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(128, 256, 3,bias=False)
        self.bn3 = nn.BatchNorm2d(256,track_running_stats=False)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(64 * 4 * 4, 128,bias=False)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, 256,bias=False)
        self.drop3 = nn.Dropout2d(p=0.5)
        self.fc3 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 64 * 4 * 4)
        # x = self.drop1(x)
        x = F.relu(self.fc1(x))
        # x = self.drop2(x)
        x = F.relu(self.fc2(x))
        # x = self.drop3(x)
        x = self.fc3(x)
        return x
