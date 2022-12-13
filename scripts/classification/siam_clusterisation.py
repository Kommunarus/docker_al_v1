import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import os
import numpy as np
from PIL import Image
from sklearn.mixture import GaussianMixture


class dataset_my(Dataset):
    def __init__(self, path_imgs, num_labels, labeled_files):
        super(dataset_my, self).__init__()
        path_to_numpy = os.path.join(path_imgs, 'numpy_siam')
        if not os.path.exists(path_to_numpy):
            os.mkdir(path_to_numpy)
        # dict_img = self.read_images(path_imgs, labeled_files)
        # data_list = []
        # for file in labeled_files:
        #     data_list.append(dict_img[file])
        # self.data = np.array(data_list)
        self.N_class = num_labels
        self.path_imgs = path_imgs
        self.labeled_files = labeled_files
        self.tran = T.Compose([T.ToPILImage(),
                          T.RandomHorizontalFlip(),
                          T.ToTensor()])


    def read_images(self, file):
        # files = os.listdir(self.path_imgs)
        # for del_dir in ['numpy_siam', 'numpy']:
        #     files.remove(del_dir)
        # dict_img = {}
        path_to_numpy = os.path.join(self.path_imgs, 'numpy_siam', file + '.npy')
        if os.path.exists(path_to_numpy):
            out = np.load(path_to_numpy)
        else:
            image = Image.open(os.path.join(self.path_imgs, file))
            image = image.resize((224, 224))

            if len(image.size) == 2:
                image = image.convert('RGB')

            np_img = np.array(image)
            np.save(path_to_numpy, np_img)

            out = np_img
        return out

    def read_labels(self, dict_id):
        labeled_data = []
        labeled_class = []
        for k, v in dict_id.items():
            labeled_data.append(k)
            labeled_class.append(v[1])

        return labeled_data, labeled_class


    def __len__(self):
        return len(self.labeled_files)

    def __getitem__(self, index):
        anchor = self.read_images(self.labeled_files[index])
        anchor = self.tran(anchor)

        return anchor

def cluster(model, device, path_to_dir, num_labels, training_data, unlabeled_data, num_sample):
    model.eval()
    # print('eval')
    train_dataset = dataset_my(path_to_dir, num_labels, training_data)
    train_loader = DataLoader(train_dataset, batch_size=10)

    data = []
    with torch.no_grad():
        for images in train_loader:
            images = images.to(device)
            image_emb = model.forward_once(images)
            data.append(image_emb.detach().cpu().numpy())

    X = np.concatenate(data)

    mix = GaussianMixture(n_components=num_labels, covariance_type='full')
    mix.fit(X)
    # print('test')
    test_dataset = dataset_my(path_to_dir, num_labels, unlabeled_data)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    # data = []
    Y_out = []
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            image_emb = model.forward_once(images)
            # data.append(image_emb.detach().cpu().numpy())
            X2 = image_emb.detach().cpu().numpy()
            # X2 = np.concatenate(data)
            Y_ = mix.predict(X2)
            Y_out.append(Y_.tolist())
    p = []
    Y_ = np.concatenate(Y_out)
    for kk in range(num_labels):
        indx = np.where(Y_ == kk)[0]
        p.append(len(indx))
    p = [x/sum(p) for x in p]
    p = [(1-x)/sum(p) for x in p]

    out = []
    # for kk in range(num_labels):
    #     indx = np.where(Y_ == kk)[0]
    #     indx_r = random.sample(indx.tolist(), k=num_sample//2)
    #     out = out + [unlabeled_data[x] for x in indx_r]
    for kk in range(num_labels):
        indx = np.where(Y_ == kk)[0]
        indx_r = random.sample(indx.tolist(), k=int(num_sample * p[kk]))
        out = out + [unlabeled_data[x] for x in indx_r]
    return out