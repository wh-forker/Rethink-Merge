import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import torch
from skimage import exposure
import cv2
from torch.utils.data import DataLoader
from datasets import load_dataset  # Huggingface's dataset library

# FILE_PATH = 'models/vis_final_model.pth'
FILE_PATH = 'models/vis_tinyimg.pth'
EPOCHS = 10
lr_decay_e = [5]
lr_decay_rate = 0.1
BATCH_SIZE = 1024


def get_images_tensor(nested_list):
    tensor_3x64x64x16 = torch.stack(
        [torch.stack([torch.stack(inner_list) for inner_list in outer_list]) for outer_list in nested_list]
    )

    tensor_16x3x64x64 = tensor_3x64x64x16.permute(3, 0, 1, 2)
    return tensor_16x3x64x64


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * 64 * 64, 200)

    def forward(self, x):
        # x = self.flatten(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def train_model(model, trainloader, criterion, optimizer, scheduler, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            images, labels = data['image'], data['label'].to(device)
            inputs = get_images_tensor(images).float().to(device)

            # Forward
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('Epoch', epoch, i, '/',  len(trainloader), 'running loss', loss.item())

        avg_loss = running_loss / len(trainloader)
        if epoch % 1 == 0:
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {avg_loss / 100:.4f}')

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch + 1}/{epochs}, Current LR: {current_lr}')
    print('Training done!')


def visualize_tensor_img(images, labels):
    images = images.cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for i in range(2):
        img = images[i].transpose((1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())
        # plt.imshow(img)
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')

    plt.show()


def test_model(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data['image'], data['label'].to(device)
            images = get_images_tensor(images).float().to(device)
            # visualize_tensor_img(images, labels)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = load_dataset('Maysee/tiny-imagenet', split='train')
    val_data = load_dataset('Maysee/tiny-imagenet', split='valid')
    test_data = load_dataset('Maysee/tiny-imagenet', split='valid')

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    def preprocess(example):
        example['image'] = transform(example['image'])
        return example


    train_data = train_data.map(preprocess)
    val_data = val_data.map(preprocess)
    test_data = test_data.map(preprocess)

    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(val_data, batch_size=2, shuffle=False)
    testloader = DataLoader(test_data, batch_size=2, shuffle=False)

    model = MLP().to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_e, gamma=lr_decay_rate)

    if not os.path.exists(FILE_PATH):
    # if True:
        train_model(model, trainloader, criterion, optimizer, scheduler, epochs=EPOCHS)
        torch.save(model.state_dict(), FILE_PATH)

    model.load_state_dict(torch.load(FILE_PATH, map_location='cpu'))
    test_model(model, testloader)

    classes = [
        "goldfish", "European fire salamander", "bullfrog", "tailed frog", "American alligator", "boa constrictor",
        "trilobite", "scorpion", "black widow", "tarantula", "centipede", "koala", "jellyfish", "brain coral", "snail",
        "sea slug", "American lobster", "spiny lobster", "black stork", "king penguin", "albatross", "dugong",
        "Yorkshire terrier", "golden retriever", "Labrador retriever", "German shepherd", "standard poodle", "tabby",
        "Persian cat", "Egyptian cat", "cougar", "lion", "brown bear", "ladybug", "grasshopper", "walking stick",
        "cockroach", "mantis", "dragonfly", "monarch", "sulphur butterfly", "sea cucumber", "guinea pig", "hog", "ox",
        "bison", "bighorn", "gazelle", "Arabian camel", "orangutan", "chimpanzee", "baboon", "African elephant",
        "lesser panda", "abacus", "academic gown", "altar", "backpack", "bannister", "barbershop", "barn", "barrel",
        "basketball", "bathtub", "beach wagon", "beacon", "beaker", "beer bottle", "bikini", "binoculars", "birdhouse",
        "bow tie", "brass", "bucket", "bullet train", "butcher shop", "candle", "cannon", "cardigan", "cash machine",
        "CD player", "chest", "Christmas stocking", "cliff dwelling", "computer keyboard", "confectionery",
        "convertible", "crane", "dam", "desk", "dining table", "dumbbell", "flagpole", "fly", "fountain", "freight car",
        "frying pan", "fur coat", "gasmask", "go-kart", "gondola", "hourglass", "iPod", "jinrikisha", "kimono",
        "lampshade", "lawn mower", "lifeboat", "limousine", "magnetic compass", "maypole", "military uniform",
        "miniskirt", "moving van", "neck brace", "obelisk", "oboe", "organ", "parking meter", "pay-phone",
        "picket fence",
        "pill bottle", "plunger", "police van", "poncho", "pop bottle", "potter's wheel", "projectile", "punching bag",
        "refrigerator", "remote control", "rocking chair", "rugby ball", "sandal", "school bus", "scoreboard",
        "sewing machine", "snorkel", "sock", "sombrero", "space heater", "spider web", "sports car",
        "steel arch bridge",
        "stopwatch", "sunglasses", "suspension bridge", "swimming trunks", "syringe", "teapot", "teddy", "thatch",
        "torch", "tractor", "triumphal arch", "trolleybus", "turnstile", "umbrella", "vestment", "viaduct",
        "volleyball",
        "water jug", "water tower", "wok", "wooden spoon", "comic book", "reel", "guacamole", "ice cream", "ice lolly",
        "goose", "drumstick", "plate", "pretzel", "mashed potato", "cauliflower", "bell pepper", "lemon", "banana",
        "pomegranate", "meat loaf", "pizza", "potpie", "espresso", "bee", "apron", "pole", "Chihuahua", "alp", "cliff",
        "coral reef", "lakeside", "seashore", "acorn", "broom", "mushroom", "nail", "chain", "slug", "orange"
    ]

    weights = model.linear.cpu().weight.data

    templates = weights.view(200, 3, 64, 64)  # put the weights into img format
    # fig, axes = plt.subplots(2, 5, figsize=(10, 5))
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    # for i in range(10):
    # for i in range(100):
    for i in range(100, 200):
        # ax = axes[i // 5, i % 5]
        # ax = axes[i // 10, i % 10]
        ax = axes[(i-100) // 10, (i-100) % 10]

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
