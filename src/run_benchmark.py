import argparse
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
import json
from glob import glob
import os
import sys
from PIL import Image as Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import models
from torchsummary import summary
from torch.optim import lr_scheduler
import copy
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmark classifier.')

    parser.add_argument('-m', '--metadata', metavar='METADATA', type=str,
                        help='Metadata file path')

    parser.add_argument('-i', '--input', metavar='INPUT DIRECTORY', type=str,
                        help='Source image directory for training the model')

    parser.add_argument('-o', '--output', metavar='MODEL DIRECTORY', type=str,
                        help='Output path for saving the model')

    parser.add_argument('-t', '--mode', metavar='MODE', type=str,
                        help='Options: [train, test]')

    args = parser.parse_args()

    model_path = args.output
    metadata_file_path = args.metadata
    face_directory = args.input
    mode = args.mode

    batch_size = 32
    num_classes = 2


    class DeepFakeDataset(Dataset):
        def __init__(self, metadata_file_path, faces, split_ratio=0.8, train=True, transform=None):
            self.is_train = train
            self.split_ratio = split_ratio
            self.transform = transform
            self.to_pil = transforms.ToPILImage()

            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            with open(metadata_file_path, 'r') as f:
                metadata = json.load(f)

                images = [Image.open(img_path) for img_path in glob(faces)]
                labels = np.array(
                    [1 if metadata[os.path.basename(face_path)] == "REAL" else 0 for face_path in glob(faces)])
                train_examples = int(len(labels) * split_ratio)

                if train:
                    self.images = images[:train_examples]
                    self.labels = labels[:train_examples]
                else:
                    self.images = images[train_examples:]
                    self.labels = labels[train_examples:]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, item):
            img = self.preprocess(self.images[item])
            label = self.labels[item]

            if self.transform:
                img = self.transform(self.to_pil(img))

            return img, label

    train_ds = DeepFakeDataset(
        metadata_file_path=metadata_file_path,
        faces=face_directory,
        train=True,
        split_ratio=0.8,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    test_ds = DeepFakeDataset(
        metadata_file_path=metadata_file_path,
        faces=face_directory,
        train=False,
        split_ratio=0.8,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=4)

    data_loaders = {
        'train': train_loader,
        'val': test_loader
    }

    dataset_sizes = {'train': len(train_ds), 'val': len(test_ds)}

    writer = SummaryWriter('runs/DFDC_benchmark')

    def train_model(model, criterion, optimizer, data_loaders, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for step, (input_samples, labels) in enumerate(data_loaders[phase]):
                    input_samples = input_samples.to(device)
                    labels = labels.to(device)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        output_samples = model(input_samples)
                        _, preds = torch.max(output_samples, 1)
                        loss = criterion(output_samples, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * input_samples.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                writer.add_scalar(f"{phase} loss", epoch_loss, epoch * dataset_sizes[phase] + step)
                writer.add_scalar(f"{phase} accuracy", epoch_acc, epoch * dataset_sizes[phase] + step)

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    # Here, we load Resnet18 pretrained weights
    model_conv = torchvision.models.resnet18(pretrained=True)

    # We need to freeze all layers except the final layer.
    for param in model_conv.parameters():
        # Set requires_grad == False to freeze the parameters so that the gradients are not computed in backward()
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    model_conv.fc = nn.Linear(model_conv.fc.in_features, 2)

    model_conv = model_conv.to(device)
    if mode == "train":

        # Set cross-entropy loss as the loss function

        criterion = nn.CrossEntropyLoss()

        # Only parameters of final layer are being optimized
        optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

        model_conv = train_model(model_conv, criterion, optimizer_conv, data_loaders, exp_lr_scheduler, num_epochs=1)

        # Save model
        torch.save(model_conv.state_dict(), model_path)

    else:
        # Load model from disk
        model_conv.load_state_dict(torch.load(model_path))

    confusion_matrix = torch.zeros(num_classes, num_classes)

    with torch.no_grad():
        for i, (inputs, classes) in enumerate(data_loaders['val']):
            inputs = inputs.to(device)
            classes = classes.to(device)
            outputs = model_conv(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    confusion_matrix = confusion_matrix.cpu().detach().numpy().astype(np.int32)
    class_names = ["Fake", "Real"]

    generate_confusion_matrix(confusion_matrix, class_names, normalize=False)
