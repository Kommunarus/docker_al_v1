from urllib import request, parse
import json
import os
import random
import matplotlib.pyplot as plt

path_labl = '/home/neptun/PycharmProjects/datasets/ds_for_docker/labels (копия)'
def al():
    url = 'http://127.0.0.1:5000/active_learning'
    params = {
        'backbone': 'mobilenet',
        'path_to_labels': '/datasets/ds_for_docker/labels (копия)',
        'path_to_img': '/datasets/ds_for_docker/train',
        'add': '100',
        'method': 'mixture2',
    }

    querystring = parse.urlencode(params)

    u = request.urlopen(url + '?' + querystring)
    resp = u.read()
    out = json.loads(resp.decode('utf-8'))['data']
    return out

def json_to_txt(files, n):
    path_to_dataset = '/home/neptun/PycharmProjects/datasets/celeba/train/'

    img0 = os.listdir(os.path.join(path_to_dataset, '0'))
    img1 = os.listdir(os.path.join(path_to_dataset, '1'))
    text_lab = ''

    for file in files:
        if file in img0:
            text_lab = text_lab + '{}\t{}\n'.format(file, 0)
        if file in img1:
            text_lab = text_lab + '{}\t{}\n'.format(file, 1)

    with open(os.path.join(path_labl, f'labels_margin_{n}.txt'), 'w') as f:
        f.write(text_lab)

def f1():
    url = 'http://127.0.0.1:5000/f1'
    params = {
        'backbone': 'mobilenet',
        'path_to_labels_train': '/datasets/ds_for_docker/labels (копия)',
        'path_to_img_train': '/datasets/ds_for_docker/train',
        'path_to_labels_val': '/datasets/ds_for_docker/labels_val.txt',
        'path_to_img_val': '/datasets/ds_for_docker/val',
    }

    querystring = parse.urlencode(params)

    u = request.urlopen(url + '?' + querystring)
    resp = u.read()
    out = json.loads(resp.decode('utf-8'))['f1_macro']
    return out

def create_file(N):
    path_to_dataset = '/home/neptun/PycharmProjects/datasets/celeba/train/'

    img0 = os.listdir(os.path.join(path_to_dataset, '0'))
    img1 = os.listdir(os.path.join(path_to_dataset, '1'))

    path_to_dataset_train = '/home/neptun/PycharmProjects/datasets/ds_for_docker/train'
    all_im = os.listdir(path_to_dataset_train)
    text_lab = ''
    val = random.sample(all_im, k=N)
    for file in val:
        if file in img0:
            text_lab = text_lab + '{}\t{}\n'.format(file, 0)
        if file in img1:
            text_lab = text_lab + '{}\t{}\n'.format(file, 1)

    with open(os.path.join(path_labl, f'labels_rnd.txt'), 'w') as f:
        f.write(text_lab)

if __name__ == '__main__':
    L = []
    for i in range(6):
        files_in_labels = os.listdir(path_labl)
        for file in files_in_labels:
            os.remove(os.path.join(path_labl, file))
        create_file(1000)
        for kk in range(20):
            step = al()
            json_to_txt(step, kk)
        f = f1()
        print(f)
        L.append(f)
    print(L)