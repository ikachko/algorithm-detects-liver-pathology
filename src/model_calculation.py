import sys
import numpy as np
import pickle

from src.img_reader import IMGReader
from src.glrlm import GLRLM
from sklearn import svm
from sklearn.externals import joblib



import pandas as pd

model_files = [
    './models/svm/svm_norma_dsh.pkl',
    './models/svm/svm_norma_gpb.pkl',
    './models/svm/svm_norma_gpc.pkl',
    './models/svm/svm_norma_vls.pkl',
    './models/svm/svm_norma_auh.pkl'
]

svm_model_files = [
    './models/svm/svm_norma_dsh.sav',
    './models/svm/svm_norma_gpb.sav',
    './models/svm/svm_norma_gpc.sav',
    './models/svm/svm_norma_vls.sav',
    './models/svm/svm_norma_auh.sav'
]

xgb_model_files = [

]

diseases = [
    'Dyscholia',
    'Hepatitis B',
    'Hepatitis C',
    'Wilson\'s Disease',
    'Autoimmune Hepatitis'
]

def load_models(model='svm'):

    models_arr = []

    model_files = None
    if model == 'svm':
        model_files = svm_model_files
    else:
        model_files = xgb_model_files

    for i in range(len(model_files)):
        models_arr.append(pickle.load(open(model_files[i], 'rb')))

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
    #
    # first_max = -1
    # first_disease = None
    #
    # indexed_predictions = []
    # for p in predictions:
    #     indexed_predictions.append()
    #
    # predictions = [predictions[i][0] + [i] for i in range(len(predictions))]
    # predictions = list(enumerate(predictions))
    # n = len(predictions)
    #
    # print(predictions)
    # for i in range(n):
    #     for j in range(0, n - i - 1):
    #         if predictions[j][0] > predictions[j + 1][0]:
    #             predictions[j][0], predictions[j + 1][0] = predictions[j + 1][0], predictions[j][0]
    #
    #
    # # predictions = [predictions[i][0:-1] for i in range(len(predictions))]
    # print(predictions)
    # for i, pred in enumerate(predictions):
    #     print("[{}] {}: True {}%, False {}%".format(
    #         i,
    #         diseases[i],
    #         int(pred[0][0] * 100),
    #         int(pred[0][1] * 100)
    #     ))
    pass



def pretty_print_labeled_predictions(labeled_predictions):

    for pred in labeled_predictions:
        print("{}: True {}%, False {}%".format(
            pred,
            int(labeled_predictions[pred][0][0] * 100),
            int(labeled_predictions[pred][0][1] * 100)
        ))
