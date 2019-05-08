import sys
import numpy as np
import pickle

from img_reader import IMGReader
from glrlm import GLRLM
from sklearn import svm
from sklearn.externals import joblib



import pandas as pd

model_files = [
    './models/svm_norma_dsh.pkl',
    './models/svm_norma_gpb.pkl',
    './models/svm_norma_gpc.pkl',
    './models/svm_norma_vls.pkl',
    './models/svm_norma_auh.pkl'
]

pickle_model_files = [
    './models/svm_norma_dsh.sav',
    './models/svm_norma_gpb.sav',
    './models/svm_norma_gpc.sav',
    './models/svm_norma_vls.sav',
    './models/svm_norma_auh.sav'
]

diseases = [
    'Dyscholia',
    'Hepatitis B',
    'Hepatitis C',
    'Wilson\'s Disease',
    'Autoimmune Hepatitis'
]

def load_models():

    models_arr = []

    for i in range(len(model_files)):
        models_arr.append(pickle.load(open(pickle_model_files[i], 'rb')))

    return models_arr


def get_features(img_arr):
    glrlm = GLRLM(img_arr)

    glrlm0 = glrlm.glrlm_0()

    return np.asarray([
        glrlm.LGRE(glrlm0),
        glrlm.HGRE(glrlm0),
        glrlm.GLNU(glrlm0)
    ])


def calculate_predictions(img_arr):
    models = load_models()
    features = get_features(img_arr)

    # print(features)
    predictions = []

    for i, model in enumerate(models):
        prediction = model.predict_proba([features])

        predictions.append(
            prediction
        )

    return predictions


def labeled_predictions(predictions):
    return {
        diseases[i]: predictions[i]

        for i in range(len(predictions))
    }

def print_sorted_predictions(predictions):
    pass


def pretty_print_labeled_predictions(labeled_predictions):

    # diseases = labeled_predictions
    # for i in range(len(labeled_predictions)):
    #     print("{}: True {}, False {}".format(
    #         ,
    #         labeled_predictions[pred][0],
    #         labeled_predictions[pred][1]
    #     ))
    # print(labeled_predictions)
    for pred in labeled_predictions:
        print("{}: True {}, False {}".format(
            pred,
            labeled_predictions[pred][0][0],
            labeled_predictions[pred][0][1]
        ))

def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        return

    img_path = sys.argv[1]

    img_arr = IMGReader.read_image(img_path)

    predictions = calculate_predictions(img_arr)

    pretty_print_labeled_predictions(labeled_predictions(predictions))


if __name__ == "__main__":
    main()