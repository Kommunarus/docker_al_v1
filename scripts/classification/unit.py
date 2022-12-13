import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

import torch
import uuid
from PIL import Image
import shutil
import random
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



data_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean,  std=std)
])


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes, n_in):
        super().__init__()

        self.fc1 = nn.Linear(n_in, 640)
        self.fc2 = nn.Linear(640, 256)
        self.fc3 = nn.Linear(256, num_classes)

        self.relu = nn.ReLU()

        self.sm = nn.Softmax(1)

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        y = self.relu(self.fc1(x))
        hid = self.relu(self.fc2(y))
        out = self.fc3(hid)
        prob = self.sm(out)

        return hid, out, prob



class Dataset_from_list(Dataset):
    def __init__(self, labeled_data, label_all, path_to_dataset_img, model_feacher_resnet, device, backbone):
        self.labeled_data = labeled_data
        self.model_feacher = model_feacher_resnet
        self.device = device
        self.dir_to_dataset = path_to_dataset_img

        le = LabelEncoder()
        le.fit(label_all)
        code_label = le.transform(label_all).tolist()
        id_label = {}
        for file, label in zip(labeled_data, label_all):
                id_label[file] = (label, code_label[label_all.index(label)])

        self.id_label = id_label
        self.transform = data_transform

        if not os.path.exists(os.path.join(path_to_dataset_img, 'numpy')):
            os.mkdir(os.path.join(path_to_dataset_img, 'numpy'))

        self.path_numpy = os.path.join(path_to_dataset_img, 'numpy', backbone)
        if not os.path.exists(self.path_numpy):
            # shutil.rmtree(path_numpy)
            os.mkdir(self.path_numpy)

    def __len__(self):
        return len(self.labeled_data)

    def __getitem__(self, idx):
        file_name = self.labeled_data[idx]
        label_Encoder = self.id_label[file_name][1]
        path_to_f = os.path.join(self.path_numpy, file_name)
        path_to_f2 = path_to_f + '.npy'
        if os.path.exists(path_to_f2):
            fea = np.load(path_to_f2)
            fea = torch.from_numpy(fea).to(self.device).to(torch.float32)
        else:
            image = Image.open(os.path.join(self.dir_to_dataset, file_name))
            image = image.resize((224, 224))

            if len(image.size) == 2:
                image = image.convert('RGB')

            np_img = np.array(image)

            if self.transform:
                image = self.transform(np_img)

            image = image.to(self.device)
            image = torch.unsqueeze(image, 0)

            fea = self.model_feacher.predict(image)
            fea = fea.squeeze()

            np.save(path_to_f, fea.tolist())

        return (fea, label_Encoder, idx)






class Dataset_transfer_learning(Dataset):
    def __init__(self, item_hidden_layers, correct_predictions, incorrect_predictions):
        self.id = [x for x in item_hidden_layers.keys()]
        self.data = [x for x in item_hidden_layers.values()]
        self.correct_predictions = correct_predictions
        self.incorrect_predictions = incorrect_predictions

    def __len__(self):
        return len(self.id)

    def __getitem__(self, item):
        data = self.data[item]
        if self.id[item] in self.correct_predictions:
            label = 1
        else:
            label = 0
        return data, label


if __name__ == '__main__':
    pass