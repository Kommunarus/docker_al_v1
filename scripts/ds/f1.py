from urllib import request, parse
import json
from scripts.classification.eval import calc_f1


def f1():
    # url = 'http://127.0.0.1:5000/f1'
    params = {
        'backbone': 'mobilenet',
        'path_to_labels_train': '/home/neptun/PycharmProjects/datasets/ds_for_docker/labels',
        'path_to_img_train': '/home/neptun/PycharmProjects/datasets/ds_for_docker/train',
        'path_to_labels_val': '/home/neptun/PycharmProjects/datasets/ds_for_docker/labels_val.txt',
        'path_to_img_val': '/home/neptun/PycharmProjects/datasets/ds_for_docker/val',
    }
    #
    # querystring = parse.urlencode(params)
    #
    # u = request.urlopen(url + '?' + querystring)
    # resp = u.read()
    # out = json.loads(resp.decode('utf-8'))['f1_macro']
    u = calc_f1(params['backbone'],
            params['path_to_img_train'],
            params['path_to_labels_train'],
            params['path_to_img_val'],
            params['path_to_labels_val'])

    return u['f1_macro']

if __name__ == '__main__':
    L = []
    for i in range(6):
        f = f1()
        print(f)
        L.append(f)
    print(L)