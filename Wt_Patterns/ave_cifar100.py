import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR100(root='../../data/cifar100/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=False)

class_avg_images = {i: torch.zeros(3, 32, 32) for i in range(100)}
class_counts = {i: 0 for i in range(100)}

for images, labels in trainloader:
    for img, label in zip(images, labels):
        class_avg_images[label.item()] += img
        class_counts[label.item()] += 1

for label in class_avg_images:
    class_avg_images[label] /= class_counts[label]
def imshow(img):
    img = img - img.min()
    img = img / img.max()
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
# fig, axes = plt.subplots(10, 10, figsize=(20, 20))

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

for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(imshow(class_avg_images[i]))
    ax.set_title(classes[i])
    ax.axis('off')

plt.tight_layout()
plt.savefig('logs/ave_cifar100.png')
plt.close()
