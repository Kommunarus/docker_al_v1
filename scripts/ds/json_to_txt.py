import os
import json
path_to_dataset = '/home/neptun/PycharmProjects/datasets/celeba/train/'

img0 = os.listdir(os.path.join(path_to_dataset, '0'))
img1 = os.listdir(os.path.join(path_to_dataset, '1'))

with open('/home/neptun/Документы/response.json') as f:
    f_txt = f.read()
val = json.loads(f_txt)
text_lab = ''
for file in val['data']:
    if file in img0:
        text_lab = text_lab + '{}\t{}\n'.format(file, 0)
    if file in img1:
        text_lab = text_lab + '{}\t{}\n'.format(file, 1)

path_to_dataset_docker = '/home/neptun/PycharmProjects/datasets/ds_for_docker/labels'
with open(os.path.join(path_to_dataset_docker, 'labels_mixture_500.txt'), 'w') as f:
    f.write(text_lab)
