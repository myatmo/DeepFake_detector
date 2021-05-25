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

from src.common import generate_confusion_matrix

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import models
from torchsummary import summary
from torch.optim import lr_scheduler
import copy
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.cuda.get_device_name()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Siamese network classifier in training/testing mode.')

    parser.add_argument('-im', '--metadata', metavar='METADATA', type=str,
                        help='Metadata file path')

    parser.add_argument('-id', '--input', metavar='INPUT DIRECTORY', type=str,
                        help='Source image directory for training the model')

    parser.add_argument('-m1', '--siamese', metavar='SIAMESE MODEL PATH', type=str,
                        help='Path for loading/saving the model')

    parser.add_argument('-m2', '--svc', metavar='SVM MODEL PATH', type=str,
                        help='Path for loading/saving the model')

    parser.add_argument('-t', '--mode', metavar='MODE', type=str,
                        help='Options: [train, test]')

    args = parser.parse_args()

    siamese_model_path = args.siamese
    svc_model_path = args.svc

    metadata_file_path = args.metadata
    face_directory = args.input
    mode = args.mode

    batch_size = 32
    num_classes = 2

    class TripletLoss(nn.Module):
        def __init__(self, margin=0.8):
            super(TripletLoss, self).__init__()
            self.margin = margin

        @staticmethod
        def calc_euclidean(x1, x2):
            return (x1 - x2).pow(2).sum(1)

        def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
            distance_positive = self.calc_euclidean(anchor, positive)
            distance_negative = self.calc_euclidean(anchor, negative)
            losses = torch.relu(distance_positive - distance_negative + self.margin)

            return losses.mean()

    class DeepFakeDataset(Dataset):
        def __init__(self, metadata_file_path, faces, split_ratio=0.8, train=True, transform=None):
            self.is_train = train
            self.split_ratio = split_ratio
            self.transform = transform
            self.to_pil = transforms.ToPILImage()
            self.image_size = (224, 224)

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
                    self.index = np.arange(0, train_examples)
                else:
                    self.images = images[train_examples:]
                    self.labels = labels[train_examples:]

        def __len__(self):
            return len(self.images)

        def __getitem__(self, item):
            anchor_img = self.preprocess(self.images[item])
            anchor_label = self.labels[item]

            if self.is_train:
                positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]

                positive_item = random.choice(positive_list)
                positive_img = self.preprocess(self.images[positive_item])

                negative_list = self.index[self.index != item][self.labels[self.index != item] != anchor_label]

                negative_item = random.choice(negative_list)
                negative_img = self.preprocess(self.images[negative_item])

                if self.transform:
                    anchor_img = self.transform(self.to_pil(anchor_img))
                    positive_img = self.transform(self.to_pil(positive_img))
                    negative_img = self.transform(self.to_pil(negative_img))

                return anchor_img, positive_img, negative_img, anchor_label

            else:
                if self.transform:
                    anchor_img = self.transform(self.to_pil(anchor_img))
                return anchor_img, anchor_label

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

    writer = SummaryWriter('runs/DFDC_siamese')

    # Define model
    n_embeddings = 64
    model = models.resnet18(pretrained=False)

    # model.fc = nn.Linear(model.fc.in_features, n_embeddings)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.PReLU(),
        nn.Linear(512, n_embeddings))

    model = torch.jit.script(model).to(device)

    def train_model(_model, criterion, _optimizer, _data_loaders, num_epochs=25, scheduler=None):
        _model.train()
        n_total_steps = dataset_sizes['train']

        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            running_loss = []

            for step, (anchor_img, positive_img, negative_img, anchor_label) in enumerate(tqdm(_data_loaders['train'], desc="Training", leave=False)):
                anchor_img = anchor_img.to(device)
                anchor_out = _model(anchor_img)

                positive_img = positive_img.to(device)
                positive_out = _model(positive_img)

                negative_img = negative_img.to(device)
                negative_out = _model(negative_img)

                _optimizer.zero_grad()

                loss = criterion(anchor_out, positive_out, negative_out)
                loss.backward()
                _optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                running_loss.append(loss.cpu().detach().numpy())

            avg_running_loss = np.mean(running_loss)
            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, num_epochs, avg_running_loss))

            # TENSORBOARD
            writer.add_scalar('Training_loss', avg_running_loss, epoch * n_total_steps + step)

        return _model

    def extract_embeddings(_model, data_loader, train=True):
        output = []
        labels = []

        _model.eval()
        with torch.no_grad():
            if train:
                for img, _, _, label in tqdm(data_loader):
                    output.append(_model(img.to(device)).cpu().numpy())
                    labels.append(label)
            else:
                for img, label in tqdm(data_loader):
                    output.append(_model(img.to(device)).cpu().numpy())
                    labels.append(label)

        output = np.concatenate(output)
        labels = np.concatenate(labels)

        # plt.figure(figsize=(15, 10), facecolor="azure")
        # for label in np.unique(labels):
        #     tmp = output[labels == label]
        #     plt.scatter(tmp[:, 0], tmp[:, 1], label=label)
        #
        # plt.legend()
        # plt.show()

        return output, labels

    if mode == "train":
        # We are optimizing all network parameters
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        # The siamese network is trained using triplet loss
        criterion_triplet_loss = torch.jit.script(TripletLoss())

        # Decay LR by a factor of 0.1 every 7 epochs
        step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        model = train_model(model, criterion_triplet_loss, optimizer, data_loaders, num_epochs=1, scheduler=None)

        # Save model
        torch.save(model.state_dict(), siamese_model_path)

        # Extract the training data embeddings from the trained CNN
        train_output, train_labels = extract_embeddings(model, train_loader, train=True)

        # Train a Support Vector Classifier
        embedding_classifier = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto'))
        embedding_classifier.fit(train_output, train_labels)

        # Save the fitted classifier to disk
        pickle.dump(embedding_classifier, open(svc_model_path, 'wb'))

    else:
        # Load models from disk
        model.load_state_dict(torch.load(siamese_model_path))

        # Load classifier
        embedding_classifier = pickle.load(open(svc_model_path, 'rb'))

    # Validation

    # 1. Extract embeddings
    embeddings_x, labels_y = extract_embeddings(model, test_loader, train=False)
    # 2. Classify embeddings
    pred_y = embedding_classifier.predict(embeddings_x)
    # 3. Evaluate predictions
    classification_score = accuracy_score(labels_y, pred_y)
    print(f"Classification score: {classification_score}")

    cf = confusion_matrix(labels_y, pred_y)
    class_names = ["Fake", "Real"]
    generate_confusion_matrix(cf, class_names, normalize=False)

