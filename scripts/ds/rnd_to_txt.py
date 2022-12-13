import os
import random
path_to_dataset = '/home/neptun/PycharmProjects/datasets/celeba/train/'

img0 = os.listdir(os.path.join(path_to_dataset, '0'))
img1 = os.listdir(os.path.join(path_to_dataset, '1'))

path_to_dataset_train = '/home/neptun/PycharmProjects/datasets/ds_for_docker/train'
all_im = os.listdir(path_to_dataset_train)
nrnd = 500
text_lab = ''
val = random.sample(all_im, k=nrnd)
for file in val:
    if file in img0:
        text_lab = text_lab + '{}\t{}\n'.format(file, 0)
    if file in img1:
        text_lab = text_lab + '{}\t{}\n'.format(file, 1)

path_to_dataset_docker = '/home/neptun/PycharmProjects/datasets/ds_for_docker/labels'
with open(os.path.join(path_to_dataset_docker, f'labels_rnd_{nrnd}.txt'), 'w') as f:
    f.write(text_lab)
