import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset  # Huggingface's dataset library


def get_images_tensor(nested_list):
    tensor_3x64x64x16 = torch.stack(
        [torch.stack([torch.stack(inner_list) for inner_list in outer_list]) for outer_list in nested_list]
    )

    tensor_16x3x64x64 = tensor_3x64x64x16.permute(3, 0, 1, 2)
    return tensor_16x3x64x64

train_data = load_dataset('Maysee/tiny-imagenet', split='train')

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
trainloader = DataLoader(train_data, batch_size=100, shuffle=False)

class_avg_images = {i: torch.zeros(3, 64, 64) for i in range(200)}
class_counts = {i: 0 for i in range(200)}

for data in trainloader:
    images, labels = data['image'], data['label']
    images = get_images_tensor(images).float()
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

fig, axes = plt.subplots(10, 10, figsize=(20, 20))

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

for i in range(100, 200):
    ax = axes[(i-100) // 10, (i-100) % 10]

    ax.imshow(imshow(class_avg_images[i]))
    ax.set_title(classes[i], fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('logs/ave_tinyimgnet_200.png')
plt.close()
