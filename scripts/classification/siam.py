import random
import numpy as np
import os
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms as T
from torch.optim.lr_scheduler import StepLR

latent_dim = 2
margin = 1

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, latent_dim),
        )
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, anchor, positive, negative):
        # get two images' features
        output1 = self.forward_once(anchor)
        output2 = self.forward_once(positive)
        output3 = self.forward_once(negative)

        return output1, output2, output3


class APP_MATCHER(Dataset):
    def __init__(self, path_imgs, num_labels, dict_id, len_ds=-1):
        super(APP_MATCHER, self).__init__()
        path_to_numpy = os.path.join(path_imgs, 'numpy_siam')
        if not os.path.exists(path_to_numpy):
            os.mkdir(path_to_numpy)
        labeled_files, labeled_class = self.read_labels(dict_id)
        dict_img = self.read_images(path_imgs, labeled_files)
        data_list = []
        for file in labeled_files:
            data_list.append(dict_img[file])
        self.data = np.array(data_list)
        self.targets = np.array([int(x) for x in labeled_class])
        self.lends = len_ds
        self.N_class = num_labels
        self.group_examples()

    def read_images(self, path_imgs, labeled_files):
        files = os.listdir(path_imgs)
        for del_dir in ['numpy_siam', 'numpy']:
            files.remove(del_dir)
        dict_img = {}
        for file in files:
            if not file in labeled_files:
                continue
            path_to_numpy = os.path.join(path_imgs, 'numpy_siam', file + '.npy')
            if os.path.exists(path_to_numpy):
                dict_img[file] = np.load(path_to_numpy)
            else:
                image = Image.open(os.path.join(path_imgs, file))
                image = image.resize((224, 224))

                if len(image.size) == 2:
                    image = image.convert('RGB')

                np_img = np.array(image)
                np.save(path_to_numpy, np_img)

                dict_img[file] = np_img
        return dict_img

    def read_labels(self, dict_id):
        labeled_data = []
        labeled_class = []
        for k, v in dict_id.items():
            labeled_data.append(k)
            labeled_class.append(v[1])

        return labeled_data, labeled_class

    def group_examples(self):
        np_arr = self.targets

        self.grouped_examples = {}
        for i in range(0, self.N_class):
            self.grouped_examples[i] = np.where((np_arr == i))[0]

    def __len__(self):
        if self.lends != -1:
            return self.lends
        return self.data.shape[0]

    def __getitem__(self, index):
        selected_class = random.randint(0, self.N_class-1)
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
        anchor = self.grouped_examples[selected_class][random_index_1]
        anchor = self.data[anchor]

        random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
        while random_index_2 == random_index_1:
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)
        positive = self.grouped_examples[selected_class][random_index_2]
        positive = self.data[positive]

        other_selected_class = random.randint(0, self.N_class-1)
        while other_selected_class == selected_class:
            other_selected_class = random.randint(0, self.N_class-1)
        random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0] - 1)
        negative = self.grouped_examples[other_selected_class][random_index_2]
        negative = self.data[negative]

        tran = T.Compose([T.ToPILImage(),
                          T.RandomHorizontalFlip(),
                          T.ToTensor()])
        anchor = tran(anchor)
        positive = tran(positive)
        negative = tran(negative)

        return anchor, positive, negative


def train(model, device, train_loader, optimizer, epoch):
    model.train()

    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    nnn = 0
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        nnn += len(anchor)
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, nnn, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    criterion = nn.TripletMarginLoss(margin=margin, p=2)

    with torch.no_grad():
        for (anchor, positive, negative) in test_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_emb, positive_emb, negative_emb = model(anchor, positive, negative)
            test_loss += criterion(anchor_emb, positive_emb, negative_emb).sum().item()  # sum up batch loss

            dist1 = (anchor_emb - positive_emb).pow(2).sum(1).sqrt()
            dist2 = (anchor_emb - negative_emb).pow(2).sum(1).sqrt()
            pred = torch.where(dist2 > dist1, 1, 0)
            correct += pred.sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(path_to_dir, device, num_labels, dict_id, epochs):
    train_kwargs = {'batch_size': 32}
    test_kwargs = {'batch_size': 32}
    # cuda_kwargs = {'num_workers': 1,
    #                'pin_memory': True,
    #                'shuffle': True}
    cuda_kwargs = {'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    # print('datasets')
    train_dataset = APP_MATCHER(path_to_dir, num_labels, dict_id, 60_000)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)

    model = SiameseNetwork().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.001)
    # print(epochs)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.97)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        scheduler.step()
    # print(model)
    return model


if __name__ == '__main__':
    # main()
    pass