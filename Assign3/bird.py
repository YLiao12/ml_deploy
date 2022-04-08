import torch
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(train_loader, valid_loader):
    for epoch in range(200):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    print('Finished Training')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    correct = 0
    total = 0
    labels_list = []
    predictions = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            for l in labels:
                labels_list.append(l)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            for p in predicted:
                predictions.append(p)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    f1 = f1_score(labels_list, predictions, average='macro' )

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    print(f'F1 score of the network on the test images: {f1} ')

    # again no gradients needed
    with torch.no_grad():
        for data in valid_loader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    PATH = './bird_model'
    torch.save(net.state_dict(), PATH)


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomResizedCrop(32),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    classes = ('Asia Brown Flycatcher', 'Blue Rock Thrush', 'Brown Shrike', 'Grey-faced Buzzard')

    train_dataset =torchvision.datasets.ImageFolder(root='./data/train',transform=transform)
    train_loader =DataLoader(train_dataset,batch_size=4, shuffle=True,num_workers=2)
    #Batch Size定义：一次训练所选取的样本数。 Batch Size的大小影响模型的优化程度和速度。
 
    valid_dataset =torchvision.datasets.ImageFolder(root='./data/val',transform=transform)
    valid_loader =DataLoader(valid_dataset,batch_size=4, shuffle=True,num_workers=2)

    train(train_loader, valid_loader)

    

    # # get some random training images
    # dataiter = iter(valid_loader)
    # images, labels = dataiter.next()
    # # show images
    # imshow(torchvision.utils.make_grid(images))
    # # print labels
    # print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))