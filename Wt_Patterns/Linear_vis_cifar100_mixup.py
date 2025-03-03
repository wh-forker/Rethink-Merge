import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch
from skimage import exposure
import cv2

FILE_PATH = 'models/[best] 20240929_cifar100_linear_sgd_vis_e100_wd-1_lrdecay.1_20e_normC_noAug.pth'
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

    ############ Create mixed weights
    mixed_weights = []
    for i in range(5):
    # for i in range(50):
        mixup_weight = 0.5 * weights[i] + 0.5 * weights[i + 5]
        # mixup_weight = 0.5 * weights[i] + 0.5 * weights[i + 50]
        mixed_weights.append(mixup_weight)

    mixed_weights = torch.stack(mixed_weights).view(5, 3, 32, 32)  # reshape to image format
    # mixed_weights = torch.stack(mixed_weights).view(50, 3, 32, 32)  # reshape to image format

    # Visualize mixed weights
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    # fig, axes = plt.subplots(5, 10, figsize=(20, 10))
    for i in range(5):
    # for i in range(50):
        ax = axes[i % 5]
        # ax = axes[i // 10, i % 10]
        template_img = mixed_weights[i].permute(1, 2, 0)
        template_img = (template_img - template_img.min()) / (template_img.max() - template_img.min())
        template_img = (template_img * 255).byte().numpy()

        ax.imshow(template_img)
        ax.set_title(f"{classes[i]} + {classes[i + 5]}", fontsize=8)
        # ax.set_title(f"{classes[i]} + {classes[i + 50]}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    # plt.savefig('logs/mixed_weights_visualization.png')
    # plt.close()
