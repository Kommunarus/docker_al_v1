import os
from torch.utils.data import DataLoader
from scripts.classification.unit import NeuralNetwork, Dataset_from_list
from scripts.classification.algorithm import least_confidence, margin_confidence, ratio_confidence, entropy_based
import random
import yaml
from autoencoder.PyTorch_VAE.models import *
from autoencoder.PyTorch_VAE.experiment import VAEXperiment
from autoencoder.PyTorch_VAE.run_remote import train_vae
from pytorch_lightning.utilities.seed import seed_everything
from autoencoder.PyTorch_VAE.dataset import VAEDataset
from sklearn.preprocessing import LabelEncoder
from scripts.classification.siam import main as trainsiam
from scripts.classification.siam_clusterisation import cluster

class Feature:
    def __init__(self, device, backbone):
        if backbone == 'b0':
            net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0',
                                          pretrained=True)
        # elif backbone == 'b4':
        #     net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4',
        #                                   pretrained=True)
        elif backbone == 'resnet50':
            net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50',
                                 weights='ResNet50_Weights.DEFAULT')
        # elif backbone == 'vgg16':
        #     net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        elif backbone == 'mobilenet':
            net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='MobileNet_V2_Weights.DEFAULT')
        else:
            net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0',
                                          pretrained=True)


        if backbone in ['b0', 'b4', '']:
            fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten()
            )
            self.old_in = net.classifier[3].in_features
            net.classifier = fc
        elif backbone in ['resnet50']:
            fc = nn.Sequential(
                nn.Flatten()
            )
            self.old_in = net.fc.in_features
            net.fc = fc
        elif backbone in ['vgg16']:
            fc = net.classifier[:4]
            self.old_in = fc[3].in_features
            net.classifier = fc
        elif backbone in ['mobilenet']:
            fc = nn.Sequential(
                nn.Flatten()
            )
            self.old_in = net.classifier[1].in_features
            net.classifier = fc
        net.eval().to(device)
        self.net = net

    def predict(self, x):
        return self.net(x)


class Feature_vae:
    def __init__(self, device, path_check, path_yaml):
        with open(path_yaml, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        model = vae_models[config['model_params']['name']](**config['model_params'])
        experiment = VAEXperiment(model,
                                  config['exp_params'])
        experiment.load_from_checkpoint(path_check, vae_model=model, params=config['exp_params'])

        experiment.model.eval().to(device)
        self.model = experiment.model

    def predict(self, x):
        return self.model(x)

    def loss_function(self, *args, **kwargs):
        return self.model.loss_function(*args, **kwargs)


def train_model(model_feacher_resnet, labeled_data, labeled_class, device, path_to_dataset_img, n_in, backbone):

    ds0 = Dataset_from_list(labeled_data, labeled_class, path_to_dataset_img, model_feacher_resnet, device, backbone)
    train_dataloader = DataLoader(ds0, batch_size=16, shuffle=True)

    loss_func = nn.CrossEntropyLoss()
    model = NeuralNetwork(num_classes=len(set(labeled_class)), n_in=n_in).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for ep in range(1, 100):
        for batch in train_dataloader:
            fea = batch[0].to(device)
            labs = batch[1].to(device)

            _, out, prob = model(fea)

            loss = loss_func(out, labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return model

def sampling_uncertainty(model, device, model_feacher, unlabeled_data, path_to_dir,
                         method='margin', num_sample=100,
                         num_labels=0, backbone='b0',):
    model.eval()
    # print('find best samples')
    dataset_train = Dataset_from_list(unlabeled_data, [0]*len(unlabeled_data), path_to_dir, model_feacher, device,
                                      backbone)
    train_dataloader = DataLoader(dataset_train, batch_size=16, shuffle=False)
    indexs = []
    values = []
    for i, batch in enumerate(train_dataloader):
        features = batch[0].to(device)
        _, out, prob = model(features)

        if method == 'least':
            confidence = least_confidence(prob, num_labels=num_labels)
        elif method == 'margin':
            confidence = margin_confidence(prob)
        elif method == 'ratio':
            confidence = ratio_confidence(prob)
        else:
            confidence = entropy_based(prob, num_labels=num_labels)

        indexs += [dataset_train.labeled_data[x] for x in batch[2].tolist()]
        values += confidence


    out_dict = {k:v for k, v in zip(indexs, values)}
    a = sorted(out_dict.items(), key=lambda x: x[1])


    temp = a[-num_sample:]
    out_name = [k for k, v in temp]
    return sorted(out_name)

def get_vae_samples(device, training_data, unlabeled_data, path_to_dir, num_sample, num_labels, dict_id, max_epochs):
    if num_labels == 2:
        out = []
        for i in range(2):
            out = out + train_vae_pod(device, training_data, unlabeled_data, path_to_dir, num_sample//2, [i], dict_id,
                                      max_epochs)
        if num_sample-len(out) > 0:
            res = list(set(unlabeled_data) - set(out))
            out = out + random.sample(res, k=num_sample-len(out))
    else:
        out = train_vae_pod(device, training_data, unlabeled_data, path_to_dir, num_sample, [], dict_id, max_epochs)
    return out

def train_vae_pod(device, training_data, unlabeled_data, path_to_dir, num_sample, filter_label, dict_id, max_epochs):
    path_yaml = '/scripts/classification/models/vae_celeba.yaml'
    path_model_vae = train_vae(path_yaml, dict_id, training_data, path_to_dir, filter_label, max_epochs)
    current_vae = Feature_vae(device, path_model_vae, path_yaml)
    with open(path_yaml, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    seed_everything(config['exp_params']['manual_seed'], True)
    config['exp_params']['M_N'] = config['exp_params']['kld_weight']
    config["data_params"]['filter_label'] = []
    config["data_params"]['limit'] = -1
    config["data_params"]['dict_id'] = dict_id
    config["data_params"]['training_data'] = unlabeled_data
    config["data_params"]['data_path'] = path_to_dir
    data = VAEDataset(**config["data_params"])
    data.setup()

    err = []
    ind = []
    train_dataset = data.train_dataloader()
    for x, l in train_dataset:
        args = current_vae.predict(x.to('cuda:0'))
        loss = current_vae.loss_function(*args, **config['exp_params'])
        err = err + loss['Reconstruction_Loss_batch']
        ind = ind + [unlabeled_data[x] for x in l.tolist()]
    out = {k: v for k, v in zip(ind, err)}
    a = sorted(out.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in a][:num_sample]


def get_mixture_samples(device, training_data, unlabeled_data, path_to_dir,
                        num_sample, num_labels, dict_id, max_epochs):
    print('train siam')
    model_siam = trainsiam(path_to_dir, device, num_labels, dict_id, max_epochs)
    out = cluster(model_siam, device, path_to_dir, num_labels, training_data, unlabeled_data, num_sample)
    return out

def for_api(backbone, method, path_to_dataset_img, add,  path_to_txt_labels):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(device)

    model_feacher_resnet = Feature(device, backbone)
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
    if 'numpy_siam' in all_items:
        all_items.remove('numpy_siam')

    unlabeled_data = list(set(all_items) - set(labeled_data))
    unlabeled_data = sorted(unlabeled_data)
    num_labels = len(set(labeled_class))

    dict_id = {}
    le = LabelEncoder()
    le.fit(labeled_class)
    code_label = le.transform(labeled_class).tolist()

    for k, v, v2 in zip(labeled_data, labeled_class, code_label):
        dict_id[k] = (v, v2)

    if method in ['margin', 'least', 'ratio', 'entropy']:
        print('train zero model', end=' ')
        model0 = train_model(model_feacher_resnet, labeled_data, labeled_class, device, path_to_dataset_img,
                             model_feacher_resnet.old_in, backbone)
        # torch.save(model0.state_dict(), f'/models/model_weights_0.pth')


    if method == 'margin':
        add_to_label_items = sampling_uncertainty(model0, device, model_feacher_resnet, unlabeled_data,
                                                  method='margin',
                                                  num_sample=add, path_to_dir=path_to_dataset_img,
                                                  num_labels=num_labels, backbone=backbone,
                                                  )
    elif method == 'least':
        add_to_label_items = sampling_uncertainty(model0, device, model_feacher_resnet, unlabeled_data,
                                                  method='least',
                                                  num_sample=add, path_to_dir=path_to_dataset_img,
                                                  num_labels=num_labels, backbone=backbone,
                                                  )
    elif method == 'ratio':
        add_to_label_items = sampling_uncertainty(model0, device, model_feacher_resnet, unlabeled_data,
                                                  method='ratio',
                                                  num_sample=add, path_to_dir=path_to_dataset_img,
                                                  num_labels=num_labels, backbone=backbone,
                                                  )
    elif method == 'entropy':
        add_to_label_items = sampling_uncertainty(model0, device, model_feacher_resnet, unlabeled_data,
                                                  method='entropy',
                                                  num_sample=add, path_to_dir=path_to_dataset_img,
                                                  num_labels=num_labels, backbone=backbone,
                                                  )
    elif method[:3] == 'vae':
        max_epochs_str = method[3:]
        if max_epochs_str == '':
            max_epochs = 500
        else:
            max_epochs = int(max_epochs_str)
        add_to_label_items = get_vae_samples(device, labeled_data, unlabeled_data,
                                                     num_labels=num_labels,
                                                     num_sample=add,
                                                     path_to_dir=path_to_dataset_img, dict_id=dict_id,
                                                     max_epochs=max_epochs)
    elif method[:7] == 'mixture':
        print('mixture')
        max_epochs_str = method[7:]
        if max_epochs_str == '':
            max_epochs = 10
        else:
            max_epochs = int(max_epochs_str)
        add_to_label_items = get_mixture_samples(device, labeled_data, unlabeled_data,
                                             num_labels=num_labels,
                                             num_sample=add,
                                             path_to_dir=path_to_dataset_img, dict_id=dict_id,
                                             max_epochs=max_epochs)
    else:
        add_to_label_items = []



    outdict = {'data': sorted(add_to_label_items)}
    return outdict


if __name__ == '__main__':
    outdict = for_api('b0', 'vae', '/home/neptun/PycharmProjects/datasets/ds_for_docker/img',
            300, '/home/neptun/PycharmProjects/datasets/ds_for_docker/labels.txt')
    print(outdict)