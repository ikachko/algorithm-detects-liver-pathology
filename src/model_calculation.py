import numpy as np
import pickle
import nrrd

import radiomics
from radiomics import featureextractor

radiomics.setVerbosity(40)

important_features = [
    'original_glrlm_GrayLevelNonUniformity',
    'original_glrlm_GrayLevelNonUniformityNormalized',
    'original_glrlm_GrayLevelVariance',
    'original_glrlm_LongRunHighGrayLevelEmphasis',
    'original_glrlm_LongRunLowGrayLevelEmphasis',
    'original_glrlm_RunEntropy',
    'original_glrlm_RunLengthNonUniformity',
    'original_glrlm_ShortRunEmphasis'
]


import pandas as pd

model_files = [
    './data/models/svm_norma_dsh.pkl',
    './data/models/svm_norma_gpb.pkl',
    './data/models/svm_norma_gpc.pkl',
    './data/models/svm_norma_vls.pkl',
    './data/models/svm_norma_auh.pkl'
]

svm_model_files = [
    './data/models/svm_norma_dsh.sav',
    './data/models/svm_norma_gpb.sav',
    './data/models/svm_norma_gpc.sav',
    './data/models/svm_norma_vls.sav',
    './data/models/svm_norma_auh.sav'
]

xgb_model_files = [
    './data/models/xgb_glrlm.pickle.dat'
]

diseases = [
    'Norma',
    'Autoimmune Hepatitis',
    'Dyscholia',
    'Hepatitis B',
    'Hepatitis C',
    'Wilson\'s Disease'
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


# def get_features(img_arr):
#     glrlm = GLRLM(img_arr)
#
#     glrlm0 = glrlm.glrlm_0()
#
#     return np.asarray([
#         glrlm.LGRE(glrlm0),
#         glrlm.HGRE(glrlm0),
#         glrlm.GLNU(glrlm0)
#     ])


def get_features(img_arr):

    data = list()

    image = img_arr[..., np.newaxis]
    label = np.ones(shape=image.shape)

    extractor = featureextractor.RadiomicsFeatureExtractor()


    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeatureClassByName('ngtdm')
    extractor.enableFeatureClassByName('gldm')


    image_path_to = './tmp/tmp_img.nrrd'
    label_path_to = './tmp/tmp_label.nrrd'

    nrrd.write(image_path_to, image)
    nrrd.write(label_path_to, label)

    result = extractor.execute(image_path_to, label_path_to)

    data.append(result)

    df = pd.DataFrame(data, dtype=np.float)

    return df[important_features]


def calculate_predictions(img_arr):
    models = load_models(model='xgb')
    features = get_features(img_arr)

    # for i, model in enumerate(models):
    #     prediction = model.predict_proba(features)
    #
    #     predictions.append(
    #         prediction
    #     )

    return models[0].predict_proba(features)


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


def pretty_print_proba_pred(predictions):
    # print(predictions[0])
    for i, pred in enumerate(predictions[0]):
        print("{}: {}%".format(
            diseases[i],
            int(pred * 100)
        ))

def pretty_print_labeled_predictions(labeled_predictions):

    for pred in labeled_predictions:
        print("{}: True {}%, False {}%".format(
            pred,
            int(labeled_predictions[pred][0][0] * 100),
            int(labeled_predictions[pred][0][1] * 100)
        ))
