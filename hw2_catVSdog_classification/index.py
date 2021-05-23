import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# 1.Get data, Generate DataLoader of train&test together
# use_gpu = torch.cuda.is_available()
use_gpu = False
path = 'data'
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
loader_batch_size = 10
transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
data_image = {x: datasets.ImageFolder(root=os.path.join(path, x),
                                      transform=transform) for x in ['train', 'test']}
data_loader = {x: DataLoader(
    dataset=data_image[x], batch_size=loader_batch_size, shuffle=True) for x in ['train', 'test']}
classes = data_image['train'].classes
classes_index = data_image['train'].class_to_idx
print('classes:', classes)
print('classes_index:', classes_index)
print('train data set:', len(data_image['train']))
print('test data set:', len(data_image['test']))


# # 2.Just a test: print classes & show images of data with size of loader_batch_size
# X_train, y_train = next(iter(data_loader['train']))
# img = torchvision.utils.make_grid(X_train)
# img = img.numpy().transpose((1, 2, 0))
# img = img*std + mean
# print([classes[i] for i in y_train])
# plt.imshow(img)
# plt.show()


# 3.Use VGG16 model and change outputs dim as 2, Define the criterion & optimizer
model = models.vgg16(pretrained=True)
for parma in model.parameters():
    parma.requires_grad = False
# print(model)
model.classifier = nn.Sequential(torch.nn.Linear(512*7*7, 4096),
                                 torch.nn.ReLU(),
                                 torch.nn.Dropout(p=0.5),
                                 torch.nn.Linear(4096, 4096),
                                 torch.nn.ReLU(),
                                 torch.nn.Dropout(p=0.5),
                                 torch.nn.Linear(4096, 2))
for index, parma in enumerate(model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True
if use_gpu:
    model = model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters())
# print(model)


# 4.Train & Test
n_epochs = 5
for epoch in range(n_epochs):
    start_time = time.time()
    print('-'*50, '\nEpoch{}/{}\n'.format(epoch+1, n_epochs), '-'*50)
    for param in ['train', 'test']:
        if param == 'train':
            model.train = True
        else:
            model.train = False

        running_loss = 0.0
        running_correct = 0
        batch = 0
        for data in data_loader[param]:
            batch += 1
            X, y = data
            if use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)
            optimizer.zero_grad()
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)
            loss = criterion(y_pred, y)
            if param == 'train':
                loss.backward()
                optimizer.step()
            running_loss += loss.data
            running_correct += torch.sum(pred == y.data)
            if batch % 10 == 0 and param == 'train':
                print('Batch {},\t Image {},\t Train Loss:{:.4f},\t Train ACC:{:.4f}%'.format(
                    batch, loader_batch_size*batch, running_loss/(loader_batch_size*batch), 100*running_correct/(loader_batch_size*batch)))

        epoch_loss = running_loss/len(data_image[param])
        epoch_correct = 100*running_correct/len(data_image[param])

        print('{} Loss:{:.4f}, Correct:{:.4f}%'.format(
            param, epoch_loss, epoch_correct))
    spend_time = time.time() - start_time
    print('Epoch{} training & testing time is: {:.0f}m {:.0f}s'.format(
        epoch+1, spend_time//60, spend_time % 60))
    torch.save(model.state_dict(), './model_params.pkl')
