import os
from sklearn.model_selection import train_test_split
import shutil

path_to_dataset = '/home/neptun/PycharmProjects/datasets/celeba/train/'
path_to_dataset_docker = '/home/neptun/PycharmProjects/datasets/ds_for_docker/'

img0 = os.listdir(os.path.join(path_to_dataset, '0'))
img1 = os.listdir(os.path.join(path_to_dataset, '1'))

allimages = img0 + img1

train, val = train_test_split(allimages, test_size=0.2, random_state=2022)
newimg, oldimg = train_test_split(train, test_size=4500, random_state=2022)

# val
text_lab = ''
for file in val:
    if file in img0:
        shutil.copyfile(os.path.join(path_to_dataset, '0', file),
                        os.path.join(path_to_dataset_docker, 'val', file))
        text_lab = text_lab + '{}\t{}\n'.format(file, 0)
    if file in img1:
        shutil.copyfile(os.path.join(path_to_dataset, '1', file),
                        os.path.join(path_to_dataset_docker, 'val', file))
        text_lab = text_lab + '{}\t{}\n'.format(file, 1)

with open(os.path.join(path_to_dataset_docker, 'labels_val.txt'), 'w') as f:
    f.write(text_lab)

# train
for file in newimg:
    if file in img0:
        shutil.copyfile(os.path.join(path_to_dataset, '0', file),
                        os.path.join(path_to_dataset_docker, 'train', file))
    if file in img1:
        shutil.copyfile(os.path.join(path_to_dataset, '1', file),
                        os.path.join(path_to_dataset_docker, 'train', file))

text_lab = ''
for file in oldimg:
    if file in img0:
        shutil.copyfile(os.path.join(path_to_dataset, '0', file),
                        os.path.join(path_to_dataset_docker, 'train', file))
        text_lab = text_lab + '{}\t{}\n'.format(file, 0)
    if file in img1:
        shutil.copyfile(os.path.join(path_to_dataset, '1', file),
                        os.path.join(path_to_dataset_docker, 'train', file))
        text_lab = text_lab + '{}\t{}\n'.format(file, 1)

with open(os.path.join(path_to_dataset_docker, 'labels.txt'), 'w') as f:
    f.write(text_lab)