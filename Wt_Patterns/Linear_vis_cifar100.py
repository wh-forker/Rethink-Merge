import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch
from skimage import exposure
import cv2

FILE_PATH = 'models/vis_final_model.pth'
EPOCHS = 100
lr_decay_e = [20, 40, 60, 80]
lr_decay_rate = 0.1


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * 32 * 32, 100)

    def forward(self, x):
        # x = self.flatten(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def train_model(model, trainloader, criterion, optimizer, scheduler, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(trainloader)
        if epoch % 1 == 0:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {avg_loss / 100:.4f}')

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{epochs}, Current LR: {current_lr}')
    print('Training done!')


def test_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])

    transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        # normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize
    ])

    # CIFAR-100 dataset
    trainset = torchvision.datasets.CIFAR100(root='../../data/cifar100/', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, pin_memory=True)
    testset = torchvision.datasets.CIFAR100(root='../../data/cifar100/', train=False, download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, pin_memory=True)

    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_e, gamma=0.1)

    if not os.path.exists(FILE_PATH):
    # if True:
        train_model(model, trainloader, criterion, optimizer, scheduler, epochs=EPOCHS)
        torch.save(model.state_dict(), FILE_PATH)

    model.load_state_dict(torch.load(FILE_PATH, map_location='cpu'))
    test_model(model, testloader)

    # visualize the weights into the shape of 3x32x32
    classes = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
        'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark',
        'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
        'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
        'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
        'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    weights = model.linear.cpu().weight.data

    templates = weights.view(100, 3, 32, 32)  # put the weights into img format
    # fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    # for i in range(10):
    for i in range(100):
        # ax = axes[i // 5, i % 5]
        ax = axes[i // 10, i % 10]

        # stretch it into 0-255
        template_img = templates[i].permute(1, 2, 0)  # (H, W, C)
        template_img = (template_img - template_img.min()) / (template_img.max() - template_img.min())
        template_img = (template_img * 255).byte().numpy()  # to 0-255

        ax.imshow(template_img)
        # ax.set_title(classes[i])
        ax.set_title(classes[i], fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    # plt.show()
    plt.savefig('logs/weights_visualization.png')
    plt.close()
