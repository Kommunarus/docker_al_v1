from urllib import request, parse
import json
import os
import random
import matplotlib.pyplot as plt

def f1():
    url = 'http://127.0.0.1:5000/f1'
    params = {
        'backbone': 'mobilenet',
        'path_to_labels_train': '/datasets/ds_for_docker/labels',
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

    path_to_dataset_docker = '/home/neptun/PycharmProjects/datasets/ds_for_docker/labels'
    with open(os.path.join(path_to_dataset_docker, f'labels_rnd.txt'), 'w') as f:
        f.write(text_lab)

if __name__ == '__main__':
    p = [3_000, 4_000]
    k = 2
    L = []
    for i in p:
        mean = 0
        m = []
        for j in range(k):
            create_file(i)
            f = f1()
            print(f)
            mean += f
            m.append(f)
        L.append(mean/k)
        plt.scatter([i]*k, m)
        # print(i, m)

    # plt.plot(p, L)
    plt.grid(True)
    plt.show()
    # print(L)