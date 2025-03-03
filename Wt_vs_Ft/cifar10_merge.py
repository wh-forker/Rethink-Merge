import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import sys
import copy

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from utils import get_model_from_sd
import argparse
from train_cifar100 import MLP, MLP_bn
import warnings
warnings.filterwarnings("ignore")
from timm import create_model
import torch.nn.functional as F


model_baseName = '20241016_vgg19_cifar10_sgd_warm1e_lr.01_gamma.2_noval_loop10'
# MODEL_NUM = 5
SEED = 999
torch.manual_seed(SEED)
np.random.seed(SEED)
# MAGNITUDE = 100


def get_arguments():
    """
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='VGG19_sgd_cifar100')
    parser.add_argument("--model", type=str, default='vgg19')
    parser.add_argument("--model_num", type=int, default=2)
    parser.add_argument("--magnitude", type=int, default=100)
    return parser


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


def test_ens_logits(model, model_pool, testloader, magnitude_on='false'):
    loaded_models = []
    for j, model_path in enumerate(model_pool):
        cur_model = copy.deepcopy(model)
        state_dict = torch.load(model_path, map_location=device)
        if magnitude_on == 'false':
            soups_state = state_dict
        elif magnitude_on == 'true':
            soups_state = {k: v * MAGNITUDE for k, v in state_dict.items()}
        elif magnitude_on == '1st_mag':
            if j == 0:
                soups_state = {k: v * MAGNITUDE for k, v in state_dict.items()}
            else:
                soups_state = state_dict
        elif magnitude_on == 'last_no_mag':
            if j == len(model_pool) - 1:
                soups_state = {k: v * MAGNITUDE for k, v in state_dict.items()}
            else:
                soups_state = state_dict
        cur_model.load_state_dict(soups_state)
        loaded_models.append(cur_model)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            logits_list = []
            for loaded_model in loaded_models:
                outputs = loaded_model(images)
                logits_list.append(outputs)
            outputs = torch.mean(torch.stack(logits_list), dim=0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')


def test_ens_fts(model, model_pool, testloader, magnitude_on='false'):
    feature_extractors = []
    classifier = None
    for j, model_path in enumerate(model_pool):
        cur_model = copy.deepcopy(model)
        state_dict = torch.load(model_path, map_location=device)
        if magnitude_on == 'false':
            soups_state = state_dict
        elif magnitude_on == 'true':
            soups_state = {k: v * MAGNITUDE for k, v in state_dict.items()}
        elif magnitude_on == '1st_mag':
            if j == 0:
                soups_state = {k: v * MAGNITUDE for k, v in state_dict.items()}
            else:
                soups_state = state_dict
        elif magnitude_on == 'last_no_mag':
            if j == len(model_pool) - 1:
                soups_state = {k: v * MAGNITUDE for k, v in state_dict.items()}
            else:
                soups_state = state_dict
        cur_model.load_state_dict(soups_state)
        feature_extractors.append(NetFeatureExtractor(cur_model))

        model_class = type(cur_model).__name__
        if j == 0:
            if model_class == 'VGG':
                classifier = cur_model.classifier[6].to(device)  # 获取第一个模型的最后一层FC
            elif model_class == 'ResNet':
                classifier = cur_model.fc.to(device)
            elif model_class == 'DenseNet':
                classifier = cur_model.classifier.to(device)
            elif 'vit_b_16' == args.model:
                classifier = cur_model.vit.heads.head.to(device)
            elif 'deit_tiny_patch16_224' == args.model or 'vit_tiny_patch16_224' == args.model:
                classifier = cur_model.vit.head.to(device)
            elif 'cnn' == args.model:
                classifier = cur_model.fc2.to(device)
            elif 'mlp' in args.model:
                classifier = cur_model.fc4.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            features_list = []
            for ft_extractor in feature_extractors:
                ft_outputs = ft_extractor(images)
                features_list.append(ft_outputs)
            ft_ave = torch.mean(torch.stack(features_list), dim=0)
            outputs = classifier(ft_ave)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')


def test_ens_fts_aveClsfier(model, model_pool, testloader, magnitude_on='false'):
    feature_extractors = []
    avg_classifier = None
    for j, model_path in enumerate(model_pool):
        cur_model = copy.deepcopy(model)
        state_dict = torch.load(model_path, map_location=device)
        if magnitude_on == 'false':
            soups_state = state_dict
        elif magnitude_on == 'true':
            soups_state = {k: v * MAGNITUDE for k, v in state_dict.items()}
        elif magnitude_on == '1st_mag':
            if j == 0:
                soups_state = {k: v * MAGNITUDE for k, v in state_dict.items()}
            else:
                soups_state = state_dict
        elif magnitude_on == 'last_no_mag':
            if j == len(model_pool) - 1:
                soups_state = {k: v * MAGNITUDE for k, v in state_dict.items()}
            else:
                soups_state = state_dict
        cur_model.load_state_dict(soups_state)
        feature_extractors.append(NetFeatureExtractor(cur_model))
        model_class = type(cur_model).__name__
        if model_class == 'VGG':
            classifier = cur_model.classifier[6].to(device)  # 获取第一个模型的最后一层FC
        elif model_class == 'ResNet':
            classifier = cur_model.fc.to(device)
        elif model_class == 'DenseNet':
            classifier = cur_model.classifier.to(device)
        elif 'vit_b_16' == args.model:
            classifier = cur_model.vit.heads.head.to(device)
        elif 'deit_tiny_patch16_224' == args.model or 'vit_tiny_patch16_224' == args.model:
            classifier = cur_model.vit.head.to(device)
        elif 'cnn' == args.model:
            classifier = cur_model.fc2.to(device)
        elif 'mlp' in args.model:
            classifier = cur_model.fc4.to(device)

        if j == 0:
            avg_classifier = copy.deepcopy(classifier).to(device)
        else:
            for param_avg, param_cur in zip(avg_classifier.parameters(), classifier.parameters()):
                param_avg.data += param_cur.data.to(device)

    for param in avg_classifier.parameters():
        param.data /= len(model_pool)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            features_list = []
            for ft_extractor in feature_extractors:
                ft_outputs = ft_extractor(images)
                features_list.append(ft_outputs)
            ft_ave = torch.mean(torch.stack(features_list), dim=0)
            outputs = avg_classifier(ft_ave)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')


class NetFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(NetFeatureExtractor, self).__init__()

        self.model_class = type(original_model).__name__
        if self.model_class == 'VGG':
            self.features = original_model.features
            self.avgpool = original_model.avgpool
            self.classifier = nn.Sequential(*list(original_model.classifier.children())[:-1])
        elif self.model_class == 'ResNet':
            self.features = nn.Sequential(
                original_model.conv1,
                original_model.bn1,
                original_model.relu,
                original_model.maxpool,
                original_model.layer1,
                original_model.layer2,
                original_model.layer3,
                original_model.layer4
            )
            self.avgpool = original_model.avgpool
        elif self.model_class == 'DenseNet':
            self.features = original_model.features
        elif 'vit_b_16' == args.model:
            self.conv = original_model.conv
            self.upsample = original_model.upsample
            self._process_input = original_model.vit._process_input
            self.class_token = original_model.vit.class_token
            self.encoder = original_model.vit.encoder
        elif 'deit_tiny_patch16_224' == args.model or 'vit_tiny_patch16_224' == args.model:
            self.conv = original_model.conv
            self.upsample = original_model.upsample
            self.vit = original_model.vit
        elif 'cnn' == args.model:
            self.conv1 = original_model.conv1
            self.conv2 = original_model.conv2
            self.conv3 = original_model.conv3
            self.fc1 = original_model.fc1
            self.dropout = original_model.dropout
        elif 'mlp' in args.model:
            self.flatten = original_model.flatten
            self.fc1 = original_model.fc1
            self.dropout1 = original_model.dropout1
            self.fc2 = original_model.fc2
            self.dropout2 = original_model.dropout2
            self.fc3 = original_model.fc3
            self.dropout3 = original_model.dropout3

    def forward(self, x):
        if self.model_class == 'VGG':
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        elif self.model_class == 'ResNet':
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        elif self.model_class == 'DenseNet':
            features = self.features(x)
            out = F.relu(features, inplace=True)
            out = F.adaptive_avg_pool2d(out, (1, 1))
            x = torch.flatten(out, 1)
        elif 'vit_b_16' == args.model:
            x = self.conv(x)
            x = self.upsample(x)
            x = self._process_input(x)
            n = x.shape[0]
            # Expand the class token to the full batch
            batch_class_token = self.class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            x = self.encoder(x)
            # Classifier "token" as used by standard language architectures
            x = x[:, 0]
        elif 'deit_tiny_patch16_224' == args.model or 'vit_tiny_patch16_224' == args.model:
            x = self.conv(x)
            x = self.upsample(x)
            x = self.vit.patch_embed(x)
            cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = self.vit.pos_drop(x + self.vit.pos_embed)
            x = self.vit.blocks(x)
            x = x[:, 0]
        elif 'cnn' == args.model:
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
        elif 'mlp' in args.model:
            x = self.flatten(x)
            x = F.relu(self.fc1(x))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = F.relu(self.fc3(x))
            x = self.dropout3(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CIFARViT(nn.Module):
    def __init__(self, model='vit_b_16'):
        super(CIFARViT, self).__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

        if model == 'vit_b_16':
            self.vit = torchvision.models.vit_b_16(pretrained=True)
            self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, 10)
        elif model == 'deit_tiny_patch16_224':
            self.vit = create_model('deit_tiny_patch16_224', pretrained=True)
            self.vit.head = nn.Linear(self.vit.head.in_features, 10)
        elif model == 'vit_tiny_patch16_224':
            self.vit = create_model('vit_tiny_patch16_224', pretrained=True)
            self.vit.head = nn.Linear(self.vit.head.in_features, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return self.vit(x)


def validate_model(model, valloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def greedy_soup(model_pool, valloader, model, testloader, magnitude_on='false'):

    best_soup = None
    best_val_acc = -float('inf')

    # Step 1: Initialize with the best single model
    for j, model_path in enumerate(model_pool):
        print(f'Testing model {j} on validation set.')
        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=device)

        cur_model = get_model_from_sd(state_dict, model)
        val_acc = validate_model(cur_model, valloader)  # Function to get validation accuracy

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_soup = state_dict  # Start with the best-performing model

    # Step 2: Greedily add models to the soup
    for j, model_path in enumerate(model_pool):
        if model_path == best_soup:
            continue  # Skip the best initial model

        print(f'Attempting to add model {j} to greedy soup.')

        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=device)

        # Create a candidate soup by adding the new model to the current best soup
        if magnitude_on == 'false':
            candidate_soup = {k: (best_soup[k] + state_dict[k]) / 2 for k in best_soup.keys()}
        elif magnitude_on == 'true':
            candidate_soup = {k: (best_soup[k] + state_dict[k] * MAGNITUDE) / 2 for k in best_soup.keys()}
        elif magnitude_on == '1st_mag':
            if j == 0:
                candidate_soup = {k: (best_soup[k] * MAGNITUDE + state_dict[k]) / 2 for k in best_soup.keys()}
            else:
                candidate_soup = {k: (best_soup[k] + state_dict[k]) / 2 for k in best_soup.keys()}
        elif magnitude_on == 'last_no_mag':
            if j == len(model_pool) - 1:
                candidate_soup = {k: (best_soup[k] + state_dict[k]) / 2 for k in best_soup.keys()}
            else:
                candidate_soup = {k: (best_soup[k] + state_dict[k] * MAGNITUDE) / 2 for k in best_soup.keys()}

        cur_model = get_model_from_sd(candidate_soup, model)
        val_acc = validate_model(cur_model, valloader)  # Evaluate on validation set

        if val_acc > best_val_acc:
            print(f'Model {j} improves validation accuracy. Adding to soup.')
            best_val_acc = val_acc
            best_soup = candidate_soup  # Update the soup with the new model

    # Final step: Test the best greedy soup on the test set
    cur_model = get_model_from_sd(best_soup, model)
    test_model(cur_model, testloader)  # Test final soup on test set


def greedy_soup_devideLen(model, valloader, model_pool, testloader, magnitude_on='false'):
    best_soup = None
    best_val_acc = -float('inf')
    model_count = len(model_pool)

    # Start from the first model
    for j, model_path in enumerate(model_pool):
        print(f'Attempting to add model {j} to greedy soup.')
        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=device)

        # Handle magnitude cases
        if magnitude_on == 'false':
            if best_soup is None:
                candidate_soup = {k: state_dict[k] for k in state_dict.keys()}
            else:
                candidate_soup = {k: (best_soup[k] * j + state_dict[k]) / (j + 1) for k in best_soup.keys()}
        elif magnitude_on == 'true':
            if best_soup is None:
                candidate_soup = {k: state_dict[k] * MAGNITUDE for k in state_dict.keys()}
            else:
                candidate_soup = {k: (best_soup[k] * j + state_dict[k] * MAGNITUDE) / (j + 1) for k in best_soup.keys()}
        elif magnitude_on == '1st_mag':
            if best_soup is None:
                candidate_soup = {k: state_dict[k] * MAGNITUDE for k in state_dict.keys()}
            else:
                candidate_soup = {k: (best_soup[k] * j + state_dict[k]) / (j + 1) for k in best_soup.keys()}
        elif magnitude_on == 'last_no_mag':
            if j == model_count - 1:
                candidate_soup = {k: (best_soup[k] * j + state_dict[k]) / (j + 1) for k in best_soup.keys()}
            elif best_soup is None:
                candidate_soup = {k: state_dict[k] * MAGNITUDE for k in state_dict.keys()}
            else:
                candidate_soup = {k: (best_soup[k] * j + state_dict[k] * MAGNITUDE) / (j + 1) for k in best_soup.keys()}

        cur_model = get_model_from_sd(candidate_soup, model)
        val_acc = validate_model(cur_model, valloader)  # Evaluate on validation set

        if val_acc > best_val_acc:
            print(f'Model {j} improves validation accuracy. Adding to soup.')
            best_val_acc = val_acc
            best_soup = candidate_soup  # Update the soup with the new model

    # Final step: Test the best greedy soup on the test set
    cur_model = get_model_from_sd(best_soup, model)
    test_model(cur_model, testloader)  # Test final soup on test set


def greedy_ens_devideLen(model, valloader, model_pool, testloader, magnitude_on='false'):
    best_soup = None
    best_val_acc = -float('inf')
    model_count = len(model_pool)
    greedy_model_pool = []

    # Start from the first model
    for j, model_path in enumerate(model_pool):
        print(f'Attempting to add model {j} to greedy soup.')
        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=device)

        # Handle magnitude cases
        if magnitude_on == 'false':
            if best_soup is None:
                candidate_soup = {k: state_dict[k] for k in state_dict.keys()}
            else:
                candidate_soup = {k: (best_soup[k] * j + state_dict[k]) / (j + 1) for k in best_soup.keys()}
        elif magnitude_on == 'true':
            if best_soup is None:
                candidate_soup = {k: state_dict[k] * MAGNITUDE for k in state_dict.keys()}
            else:
                candidate_soup = {k: (best_soup[k] * j + state_dict[k] * MAGNITUDE) / (j + 1) for k in best_soup.keys()}
        elif magnitude_on == '1st_mag':
            if best_soup is None:
                candidate_soup = {k: state_dict[k] * MAGNITUDE for k in state_dict.keys()}
            else:
                candidate_soup = {k: (best_soup[k] * j + state_dict[k]) / (j + 1) for k in best_soup.keys()}
        elif magnitude_on == 'last_no_mag':
            if j == model_count - 1:
                candidate_soup = {k: (best_soup[k] * j + state_dict[k]) / (j + 1) for k in best_soup.keys()}
            elif best_soup is None:
                candidate_soup = {k: state_dict[k] * MAGNITUDE for k in state_dict.keys()}
            else:
                candidate_soup = {k: (best_soup[k] * j + state_dict[k] * MAGNITUDE) / (j + 1) for k in best_soup.keys()}

        cur_model = get_model_from_sd(candidate_soup, model)
        val_acc = validate_model(cur_model, valloader)  # Evaluate on validation set

        if val_acc > best_val_acc:
            print(f'Model {j} improves validation accuracy. Adding to soup.')
            best_val_acc = val_acc
            best_soup = candidate_soup  # Update the soup with the new model
            greedy_model_pool.append(model_path)

    # Final step: Test the best greedy soup on the test set
    # cur_model = get_model_from_sd(best_soup, model)
    # test_model(cur_model, testloader)  # Test final soup on test set

    print('----------------------------------------')
    print('Evaluate Logits Ensemble models w/wo magnitudes...')
    print('magnitude_on=false')
    test_ens_logits(model, greedy_model_pool, testloader, magnitude_on='false')
    print('magnitude_on=true')
    test_ens_logits(model, greedy_model_pool, testloader, magnitude_on='true')
    print('magnitude_on=1st_mag')
    test_ens_logits(model, greedy_model_pool, testloader, magnitude_on='1st_mag')
    print('magnitude_on=last_no_mag')
    test_ens_logits(model, greedy_model_pool, testloader, magnitude_on='last_no_mag')

    print('----------------------------------------')
    print('Evaluate Features Ensemble models w/wo magnitudes, classify with averaged classifier...')
    print('magnitude_on=false')
    test_ens_fts_aveClsfier(model, greedy_model_pool, testloader, magnitude_on='false')
    print('magnitude_on=true')
    test_ens_fts_aveClsfier(model, greedy_model_pool, testloader, magnitude_on='true')
    print('magnitude_on=1st_mag')
    test_ens_fts_aveClsfier(model, greedy_model_pool, testloader, magnitude_on='1st_mag')
    print('magnitude_on=last_no_mag')
    test_ens_fts_aveClsfier(model, greedy_model_pool, testloader, magnitude_on='last_no_mag')


if __name__ == '__main__':
    parser = get_arguments()
    print(parser)
    args = parser.parse_args()
    MODEL_NUM = args.model_num
    MAGNITUDE = args.magnitude

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalize = transforms.Normalize(mean=[0.5071, 0.4865, 0.4409],
                                     std=[0.2673, 0.2564, 0.2762])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='../../data/cifar10/', train=True, download=True, transform=None)
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(trainset, [train_size, val_size])
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='../../data/cifar10/', train=False, download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, pin_memory=True)

    if args.model == 'vgg11':
        model = torchvision.models.vgg11(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'vgg11_bn':
        model = torchvision.models.vgg11_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'vgg13':
        model = torchvision.models.vgg13(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'vgg13_bn':
        model = torchvision.models.vgg13_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'vgg16_bn':
        model = torchvision.models.vgg16_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'vgg19_bn':
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 10)
    elif args.model == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'resnet152':
        model = torchvision.models.resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif args.model == 'densenet121':
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 10)
    elif args.model == 'densenet161':
        model = torchvision.models.densenet161(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 10)
    elif args.model == 'densenet169':
        model = torchvision.models.densenet169(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 10)
    elif args.model == 'densenet201':
        model = torchvision.models.densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 10)
    elif args.model == 'vit_b_16':
        model = CIFARViT(model='vit_b_16')
    elif args.model == 'deit_tiny_patch16_224':
        model = CIFARViT(model='deit_tiny_patch16_224')
    elif args.model == 'vit_tiny_patch16_224':
        model = CIFARViT(model='vit_tiny_patch16_224')
    elif args.model == 'mlp':
        model = MLP()
    elif args.model == 'mlp_bn':
        model = MLP_bn()
    elif args.model == 'cnn':
        model = SimpleCNN()
    model = model.to(device)

    print('Soup MODEL_NUM:', MODEL_NUM)
    model_pool = []
    for i in range(MODEL_NUM):
        model_pool.append('models/' + model_baseName + '_' + str(i+1) + '_best.pth')

    # 1. Evaluate individual models
    print('----------------------------------------')
    print('1. Evaluate individual models...')
    for name in model_pool:
        cur_model = copy.deepcopy(model)
        cur_model.load_state_dict(torch.load(name, map_location=device))
        test_model(cur_model, testloader)

    # 2. Evaluate Uniform Soup models
    print('----------------------------------------')
    print('2. Evaluate Uniform Soup models...')
    for j, name in enumerate(model_pool):
        NUM_MODELS = len(model_pool)
        model_path = name
        print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')

        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=device)
        if j == 0:
            uniform_soup = {k: v * (1. / NUM_MODELS) for k, v in state_dict.items()}
        else:
            uniform_soup = {k: v * (1. / NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

    cur_model = get_model_from_sd(uniform_soup, model)
    test_model(cur_model, testloader)

    # 3. Evaluate the MAGNITUDEx model
    print('----------------------------------------')
    print('3. Evaluate the MAGNITUDEx model...')
    print('MAGNITUDE is', MAGNITUDE)
    for name in model_pool:
        model_path = name
        state_dict = torch.load(model_path, map_location=device)
        ten_times = {k: v * MAGNITUDE for k, v in state_dict.items()}
        cur_model = get_model_from_sd(ten_times, model)
        test_model(cur_model, testloader)

    # 4. Evaluate Uniform Soup models after all merging MAGNITUDEx models
    print('----------------------------------------')
    print('4. Evaluate Uniform Soups after merging MAGNITUDEx models...')
    for j, name in enumerate(model_pool):
        NUM_MODELS = len(model_pool)
        model_path = name
        print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')

        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=device)
        if j == 0:
            uniform_soup = {k: v * (1. / NUM_MODELS) * MAGNITUDE for k, v in state_dict.items()}
        else:
            uniform_soup = {k: v * (1. / NUM_MODELS) * MAGNITUDE + uniform_soup[k] for k, v in state_dict.items()}

    cur_model = get_model_from_sd(uniform_soup, model)
    test_model(cur_model, testloader)


    # 5. Evaluate Uniform Soup models after merging MAGNITUDEx models
    print('----------------------------------------')
    print('5. Evaluate Uniform Soups after merging 1st MAGNITUDEx and Non MAGNITUDEx models...')
    for j, name in enumerate(model_pool):
        NUM_MODELS = len(model_pool)
        model_path = name
        print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')

        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=device)
        if j == 0:
            uniform_soup = {k: v * (1. / NUM_MODELS) * MAGNITUDE for k, v in state_dict.items()}
        else:
            uniform_soup = {k: v * (1. / NUM_MODELS) + uniform_soup[k] for k, v in state_dict.items()}

    cur_model = get_model_from_sd(uniform_soup, model)
    test_model(cur_model, testloader)

    # 6. Evaluate Uniform Soup models after merging MAGNITUDEx models
    print('----------------------------------------')
    print('6. Evaluate Uniform Soups after merging Non MAGNITUDEx and last MAGNITUDEx models...')
    for j, name in enumerate(model_pool):
        NUM_MODELS = len(model_pool)
        model_path = name
        print(f'Adding model {j} of {NUM_MODELS - 1} to uniform soup.')

        assert os.path.exists(model_path)
        state_dict = torch.load(model_path, map_location=device)
        if j == 0:
            uniform_soup = {k: v * (1. / NUM_MODELS) for k, v in state_dict.items()}
        else:
            uniform_soup = {k: v * (1. / NUM_MODELS) * MAGNITUDE + uniform_soup[k] for k, v in state_dict.items()}

    cur_model = get_model_from_sd(uniform_soup, model)
    test_model(cur_model, testloader)

    # 7. Evaluate Ensemble models averaging on Logits
    print('----------------------------------------')
    print('7. Evaluate Ensemble models averaging on Logits...')
    test_ens_logits(model, model_pool, testloader)

    # 8. Evaluate Ensemble models averaging on Features
    print('----------------------------------------')
    print('8. Evaluate Ensemble models averaging on Features...')
    test_ens_fts(model, model_pool, testloader)

    # 9. Evaluate Ensemble models averaging on Features, classify with averaged classifier
    print('----------------------------------------')
    print('9. Evaluate Ensemble models averaging on Features, classify with averaged classifier...')
    test_ens_fts_aveClsfier(model, model_pool, testloader)

    # 10. greedy soups (requires an extra val set)
    print('----------------------------------------')
    print('10. greedy soups...')
    greedy_soup(model_pool, valloader, model, testloader)

    # 11. greedy soups for all MAGNITUDEx models
    print('----------------------------------------')
    print('11. greedy soups for all MAGNITUDEx models...')
    greedy_soup(model_pool, valloader, model, testloader, magnitude_on='true')

    # 12. greedy soups for only 1st MAGNITUDEx
    print('----------------------------------------')
    print('12. greedy soups for only 1st MAGNITUDEx...')
    greedy_soup(model_pool, valloader, model, testloader, magnitude_on='1st_mag')

    # 13. greedy soups for only last not MAGNITUDEx model
    print('----------------------------------------')
    print('13. greedy soups for only last not MAGNITUDEx model...')
    greedy_soup(model_pool, valloader, model, testloader, magnitude_on='last_no_mag')

    # 14. Evaluate Logits Ensemble models with magnitudes
    print('----------------------------------------')
    print('14. Evaluate Logits Ensemble models with magnitudes...')
    print('magnitude_on=true')
    test_ens_logits(model, model_pool, testloader, magnitude_on='true')
    print('magnitude_on=1st_mag')
    test_ens_logits(model, model_pool, testloader, magnitude_on='1st_mag')
    print('magnitude_on=last_no_mag')
    test_ens_logits(model, model_pool, testloader, magnitude_on='last_no_mag')

    # 15. Evaluate Features Ensemble models with magnitudes, classify with averaged classifier
    print('----------------------------------------')
    print('15. Evaluate Features Ensemble models with magnitudes, classify with averaged classifier...')
    print('magnitude_on=true')
    test_ens_fts_aveClsfier(model, model_pool, testloader, magnitude_on='true')
    print('magnitude_on=1st_mag')
    test_ens_fts_aveClsfier(model, model_pool, testloader, magnitude_on='1st_mag')
    print('magnitude_on=last_no_mag')
    test_ens_fts_aveClsfier(model, model_pool, testloader, magnitude_on='last_no_mag')

    # 16. greedy soup deviding by length of models
    print('----------------------------------------')
    print('16. greedy soup deviding by length of models...')
    print('magnitude_on=false')
    greedy_soup_devideLen(model, valloader, model_pool, testloader, magnitude_on='false')
    print('magnitude_on=true')
    greedy_soup_devideLen(model, valloader, model_pool, testloader, magnitude_on='true')
    print('magnitude_on=1st_mag')
    greedy_soup_devideLen(model, valloader, model_pool, testloader, magnitude_on='1st_mag')
    print('magnitude_on=last_no_mag')
    greedy_soup_devideLen(model, valloader, model_pool, testloader, magnitude_on='last_no_mag')

    # 17. Evaluate Features Ensemble models with magnitudes
    print('----------------------------------------')
    print('17. Evaluate Features Ensemble models with magnitudes...')
    print('magnitude_on=true')
    test_ens_fts(model, model_pool, testloader, magnitude_on='true')
    print('magnitude_on=1st_mag')
    test_ens_fts(model, model_pool, testloader, magnitude_on='1st_mag')
    print('magnitude_on=last_no_mag')
    test_ens_fts(model, model_pool, testloader, magnitude_on='last_no_mag')

    # 18. Evaluate Greedy Ensemble models
    print('----------------------------------------')
    print('18. Evaluate Greedy Ensemble models...')
    greedy_ens_devideLen(model, valloader, model_pool, testloader)
