from scripts.classification.train import train_model, Feature
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import os
from scripts.classification.unit import NeuralNetwork, Dataset_from_list


def eval_model(model_feacher_resnet, model, labeled_data, labeled_class, device, path_to_dataset_img, n_in, backbone):

    ds0 = Dataset_from_list(labeled_data, labeled_class, path_to_dataset_img, model_feacher_resnet, device, backbone)
    val_dataloader = DataLoader(ds0, batch_size=16, shuffle=False)
    model.eval()
    y_true = []
    y_pred = []

    for batch in val_dataloader:
        fea = batch[0].to(device)
        labs = batch[1].to(device)
        _, _, prob = model(fea)

        y_true = y_true + labs.tolist()
        pred = torch.argmax(prob, 1).tolist()
        y_pred = y_pred + pred
    acc = f1_score(y_true, y_pred, average='macro')
    return acc

def calc_f1(backbone, path_to_dataset_img, path_to_txt_labels, path_to_dataset_img_val, path_to_txt_labels_val):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_feacher_resnet = Feature(device, backbone)

    # train
    labeled_data = []
    labeled_class = []
    if os.path.isfile(path_to_txt_labels):
        with open(path_to_txt_labels) as f:
            for row in f.readlines():
                row_split = row.strip().split('\t')
                labeled_data.append(row_split[0])
                labeled_class.append(row_split[1])
    if os.path.isdir(path_to_txt_labels):
        all_lab = os.listdir(path_to_txt_labels)
        for file in all_lab:
            with open(os.path.join(path_to_txt_labels, file)) as f:
                for row in f.readlines():
                    row_split = row.strip().split('\t')
                    labeled_data.append(row_split[0])
                    labeled_class.append(row_split[1])

    all_items = os.listdir(path_to_dataset_img)
    if 'numpy' in all_items:
        all_items.remove('numpy')

    dict_id = {}
    le = LabelEncoder()
    le.fit(labeled_class)
    code_label = le.transform(labeled_class).tolist()

    for k, v, v2 in zip(labeled_data, labeled_class, code_label):
        dict_id[k] = (v, v2)

    model0 = train_model(model_feacher_resnet, labeled_data, labeled_class, device, path_to_dataset_img,
                         model_feacher_resnet.old_in, backbone)

    # val
    labeled_data_val = []
    labeled_class_val = []
    with open(path_to_txt_labels_val) as f:
        for row in f.readlines():
            row_split = row.strip().split('\t')
            labeled_data_val.append(row_split[0])
            labeled_class_val.append(row_split[1])

    all_items = os.listdir(path_to_dataset_img_val)
    if 'numpy' in all_items:
        all_items.remove('numpy')

    dict_id = {}
    code_label = le.transform(labeled_class_val).tolist()

    for k, v, v2 in zip(labeled_data_val, labeled_class_val, code_label):
        dict_id[k] = (v, v2)

    f1 = eval_model(model_feacher_resnet, model0, labeled_data_val, labeled_class_val, device, path_to_dataset_img_val,
                         model_feacher_resnet.old_in, backbone)

    outdict = {'f1_macro': f1}

    return outdict