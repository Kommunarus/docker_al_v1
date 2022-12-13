from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
from scripts.classification.train import for_api
from scripts.classification.eval import calc_f1


app = Flask(__name__)
api = Api(app)

def make_lab(backbone, method, path_to_dataset_img, add, path_to_txt_labels):
    out = for_api(backbone, method, path_to_dataset_img, add, path_to_txt_labels)
    return jsonify(out)

def make_f1(backbone, path_to_dataset_img, path_to_txt_labels, path_to_dataset_img_val, path_to_txt_labels_val):
    out = calc_f1(backbone, path_to_dataset_img, path_to_txt_labels, path_to_dataset_img_val, path_to_txt_labels_val)
    return jsonify(out)

class active_learning(Resource):
    @staticmethod
    def get():
        backbone = reqparse.request.args['backbone']
        add = int(reqparse.request.args['add'])
        method = reqparse.request.args['method']
        path_to_txt_labels = reqparse.request.args['path_to_labels']
        path_to_dataset_img = reqparse.request.args['path_to_img']
        return make_lab(backbone, method, path_to_dataset_img, add, path_to_txt_labels)

class f1(Resource):
    @staticmethod
    def get():
        backbone = reqparse.request.args['backbone']

        path_to_txt_labels = reqparse.request.args['path_to_labels_train']
        path_to_dataset_img = reqparse.request.args['path_to_img_train']

        path_to_txt_labels_val = reqparse.request.args['path_to_labels_val']
        path_to_dataset_img_val = reqparse.request.args['path_to_img_val']

        return make_f1(backbone,
                       path_to_dataset_img, path_to_txt_labels,
                       path_to_dataset_img_val, path_to_txt_labels_val, )


api.add_resource(active_learning, '/active_learning')
api.add_resource(f1, '/f1')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
