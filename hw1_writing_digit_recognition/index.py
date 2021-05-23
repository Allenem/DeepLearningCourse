import os
import gzip
import struct
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader


# 1.1.Function to read label & image
def _read(image, label):
    # minist_dir = os.path.dirname(__file__)+'/MNIST_data/'
    minist_dir = './MNIST_data/'
    with gzip.open(minist_dir + label) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(minist_dir + image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(
            len(label), rows, cols)
    return image, label


# 1.2.Function to get data from .gz file
def get_data():
    train_img, train_label = _read(
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz')

    test_img, test_label = _read(
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz')
    return [train_img, train_label, test_img, test_label]


# 1.3.LeNet5
# 32-5+1=28,(28-2)/2+1=14,14-5+1=10,(10-2)/2+1=5,5-5+1=1,
# 1*120 -> 84=7*12 -> 10
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(torch.tanh(self.conv1(x)), (2, 2))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.max_pool2d(torch.tanh(self.conv2(x)), (2, 2))
        x = F.dropout(x, p=0.3, training=self.training)
        x = x.view(-1, self.num_flat_features(x))
        # print('x.size:', x.size())  # [100, 400]
        x = torch.tanh(self.fc1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = torch.tanh(self.fc2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc3(x)
        return x

    # Flatten the size of x (BATCH_SIZE*16*5*5 -> BATCH_SIZE*400)
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# 1.4.Function to initialize parameters
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weigth.data.fill_(1)
        m.bias.data.zero_()


# 1.5.Function to train network
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target.long())
        optimizer.zero_grad()
        outputs = model(data)
        # print(data.shape, outputs.shape, target.shape)  # [100, 1, 28, 28] [100, 10] [100]
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if (batch_idx+1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), loss.data))


# 1.6.Function to test network
def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data = Variable(data)
        # data = Variable(data, volatile=True)
        target = Variable(target.long())
        outputs = model(data)
        test_loss += criterion(outputs, target).data
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 2.1.Set some parameters
# use_gpu = torch.cuda.is_available()
use_gpu = False
BATCH_SIZE = 100
kwargs = {'num_workers': 0, 'pin_memory': True}


# 2.2.Prepare data of train_img, train_label, test_img, test_label
train_img, train_label, test_img, test_label = get_data()
train_x, train_y = torch.from_numpy(
    train_img.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(train_label.astype(int))
test_x, test_y = torch.from_numpy(
    test_img.reshape(-1, 1, 28, 28)).float(), torch.from_numpy(test_label.astype(int))
train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True, batch_size=BATCH_SIZE, **kwargs)
test_loader = DataLoader(dataset=test_dataset,
                         shuffle=True, batch_size=BATCH_SIZE, **kwargs)
# print(len(train_loader), len(test_loader)) # 600 100


# # 2.3.Print the shape(100*1*28*28) of data & labels(100) in each batch , and show the data as gray img
# for i, (data, target) in enumerate(train_loader):
#     if i == 0:
#         print(data.shape, target)
#         for j in range(BATCH_SIZE):
#             if j < 10:
#                 plt.figure()
#                 plt.imshow(data[j][0], cmap='gray')
#                 plt.show()


# 2.4.Instantiation network, set optimizer & criterion, apply weight_init funcction
model = LeNet5()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
criterion = nn.CrossEntropyLoss(size_average=False)
model.apply(weight_init)


# 2.5.Train & test
ENDEPOCH = 99
for epoch in range(0, ENDEPOCH+1):
    print('----------------start train-----------------')
    train(epoch)
    print('----------------end train-----------------')
    # save the parameters of final model
    if epoch == ENDEPOCH:
        torch.save(model.state_dict(), './model_params.pkl')
    # test each epoch
    print('----------------start test-----------------')
    test()
    print('----------------end test-----------------')

# load the parameters of final model and then test the model
model = LeNet5()
model.load_state_dict(torch.load('./model_params.pkl'))
print('----------------start final test-----------------')
test()
print('----------------end final test-----------------')
