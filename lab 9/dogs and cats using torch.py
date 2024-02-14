import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_images(path):
    images = []
    filenames = os.listdir(path)

    for filename in tqdm(filenames):
        image = cv2.imread(os.path.join(path, filename))
        image = cv2.resize(image, dsize=(100, 100))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    return np.array(images)


cats_train = load_images("E:/KLU/3rd year/3_2/deep learning/Deep Learning Programs/cats_and_dog/train/cats")
dogs_train = load_images("E:/KLU/3rd year/3_2/deep learning/Deep Learning Programs/cats_and_dog/train/dogs")

cats_test = load_images("E:/KLU/3rd year/3_2/deep learning/Deep Learning Programs/cats_and_dog/test/cats")
dogs_test = load_images("E:/KLU/3rd year/3_2/deep learning/Deep Learning Programs/cats_and_dog/test/dogs")

X_train = np.append(cats_train, dogs_train, axis=0)
X_test = np.append(cats_test, dogs_test, axis=0)



y_train = np.array([0] * len(cats_train) + [1] * len(dogs_train))
y_test = np.array([0] * len(cats_test) + [1] * len(dogs_test))


def show_images(images, labels, start_index):
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20, 12))

    counter = start_index

    for i in range(4):
        for j in range(8):
            axes[i, j].set_title(labels[counter].item())
            axes[i, j].imshow(images[counter], cmap='gray')
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)
            counter += 1
    plt.show()


print(y_train.shape)
print(y_test.shape)


def show_images(images, labels, start_index):
    fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20, 12))

    counter = start_index

    for i in range(4):
        for j in range(8):
            axes[i, j].set_title(labels[counter].item())
            axes[i, j].imshow(images[counter], cmap='gray')
            axes[i, j].get_xaxis().set_visible(False)
            axes[i, j].get_yaxis().set_visible(False)
            counter += 1
    plt.show()


show_images(X_train, y_train, 0)

print(y_train[:10])

y_train = torch.from_numpy(y_train.reshape(len(y_train), 1))
y_test = torch.from_numpy(y_test.reshape(len(y_test), 1))

print(y_train[:10])

transforms_train = transforms.Compose([transforms.ToTensor(),  # convert to tensor
                                       transforms.RandomRotation(degrees=20),
                                       transforms.RandomHorizontalFlip(p=0.5),
                                       transforms.RandomVerticalFlip(p=0.005),
                                       transforms.RandomGrayscale(p=0.2),
                                       transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                       # squeeze to -1 and 1
                                       ])
transforms_test = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])



class Cat_Dog_Dataset():
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return (image, label)


train_dataset = Cat_Dog_Dataset(images=X_train, labels=y_train, transform=transforms_train)
test_dataset = Cat_Dog_Dataset(images=X_test, labels=y_test, transform=transforms_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True)

iterator = iter(train_loader)
image_batch, label_batch = next(iterator)

print(image_batch.shape)

image_batch_permuted = image_batch.permute(0, 2, 3, 1)

print(image_batch_permuted.shape)

show_images(image_batch_permuted, label_batch, 0)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn0 = nn.BatchNorm2d(num_features=16)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        # self.maxpool

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        # self.maxpool

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        # self.maxpool

        self.dropout = nn.Dropout(p=0.5)
        self.fc0 = nn.Linear(in_features=128 * 6 * 6, out_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = x.reshape(x.shape[0], -1)

        x = self.dropout(x)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return x


model = CNN().to(device)

summary(model, input_size=(3, 100, 100))


loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def predict_test_data(model, test_loader):
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for batch, (X_test, y_test) in enumerate(test_loader):
            X_test = X_test.float().to(device)
            y_test = y_test.float().to(device)

            # Calculate loss (forward propagation)
            test_preds = model(X_test)
            test_loss = loss_function(test_preds, y_test)

            # Calculate accuracy
            rounded_test_preds = torch.round(test_preds)
            num_correct += torch.sum(rounded_test_preds == y_test)
            num_samples += len(y_test)

    model.train()

    test_acc = num_correct / num_samples

    return test_loss, test_acc


train_losses = []  # Training and testing loss was calculated based on the last batch of each epoch.
test_losses = []
train_accs = []
test_accs = []

for epoch in range(5):

    num_correct_train = 0
    num_samples_train = 0
    for batch, (X_train, y_train) in tqdm(enumerate(train_loader), total=len(train_loader)):
        X_train = X_train.float().to(device)
        y_train = y_train.float().to(device)

        # Forward propagation
        train_preds = model(X_train)
        train_loss = loss_function(train_preds, y_train)

        # Calculate train accuracy
        with torch.no_grad():
            rounded_train_preds = torch.round(train_preds)
            num_correct_train += torch.sum(rounded_train_preds == y_train)
            num_samples_train += len(y_train)

        # Backward propagation
        optimizer.zero_grad()
        train_loss.backward()

        # Gradient descent
        optimizer.step()

    train_acc = num_correct_train / num_samples_train
    test_loss, test_acc = predict_test_data(model, test_loader)

    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    train_accs.append(train_acc.item())
    test_accs.append(test_acc.item())

    print(f'Epoch: {epoch} \t|' \
          f' Train loss: {np.round(train_loss.item(), 3)} \t|' \
          f' Test loss: {np.round(test_loss.item(), 3)} \t|' \
          f' Train acc: {np.round(train_acc.item(), 2)} \t|' \
          f' Test acc: {np.round(test_acc.item(), 2)}')

plt.figure(figsize=(10, 6))
plt.grid()
plt.plot(train_losses)
plt.plot(test_losses)
plt.legend(['train_losses', 'test_losses'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


iter_test = iter(test_loader)
img_test, lbl_test = next(iter_test)

# Predict labels
preds_test = model(img_test.to(device))
img_test_permuted = img_test.permute(0, 2, 3, 1)
rounded_preds = preds_test.round()

# Show test images and the predicted labels
show_images(img_test_permuted, rounded_preds, 0)
