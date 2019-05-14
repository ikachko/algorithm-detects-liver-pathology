import numpy as np
import pickle
import nrrd
import radiomics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from radiomics import featureextractor

from .globals import (
    SVM_MODEL_FILES,
    XGB_MODEL_FILES,
    IMPORTANT_FEATURES,
    X_FEATURES,
    DISEASES
)
radiomics.setVerbosity(40)

dataset = './data/datasets/radiomics_features.csv'


def load_models(model='svm'):

    models_arr = []

    model_files = None
    if model == 'svm':
        model_files = SVM_MODEL_FILES
    else:
        model_files = XGB_MODEL_FILES

    for i in range(len(model_files)):
        models_arr.append(pickle.load(open(model_files[i], 'rb')))

    return models_arr


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

    return df[X_FEATURES]


def read_dataset(dataset, sep):
    return pd.read_csv(dataset, sep)[IMPORTANT_FEATURES]


def labeled_predictions(predictions):
    return {
        DISEASES[i]: predictions[i]

        for i in range(len(predictions))
    }


def pretty_print_proba_pred(predictions):
    # print(predictions[0])
    for i, pred in enumerate(predictions[0]):
        print("{}: {}%".format(
            DISEASES[i],
            int(pred * 100)
        ))


def pretty_print_labeled_predictions(labeled_predictions):

    for pred in labeled_predictions:
        print("{}: True {}%, False {}%".format(
            pred,
            int(labeled_predictions[pred][0][0] * 100),
            int(labeled_predictions[pred][0][1] * 100)
        ))

def compared_str(new, norma, feature):
    if new > norma:
        return "".join(['New ', str(new), ' > ', 'Norma ', str(norma), ', ', feature, ' ', str(float(new)/norma * 100), ' % more'])
    else:
        return "".join(['New ', str(new), ' < ', 'Norma ', str(norma), ', ', feature, ' ', str(float(new)/norma * 100), ' % less'])


def visualize_difference(new, norma):

    for i, feature in enumerate(new.columns):
        ax = plt.subplot(3, 3, i + 1)
        plt.scatter(1, new.iloc[:, i].values[0])
        plt.scatter(1, norma.iloc[:, i].values[0])
        plt.legend(['New', 'Norma'])

    plt.show()


def calculate_predictions(img_arr):
    models = load_models(model='xgb')
    features = get_features(img_arr)

    norma_dataset = read_dataset(dataset, sep=';')

    norma_dataset = norma_dataset[norma_dataset['diagnosis_code'] == 0].iloc[:, :-1].describe().loc[['mean'], :]
    # print(norma_dataset)
    print(features.iloc[:, 0].values)

    stat_features = features.describe().loc[['mean'], :]

    norma_dataset['isNorma'] = 'Norma'
    stat_features['isNorma'] = 'New'

    one_dataset = pd.concat([norma_dataset, stat_features])

    # sns.countplot(data=one_dataset, x='isNorma', hue='isNorma')
    # plt.show()
    # print(one_dataset)

    # for i, feature in enumerate(stat_features.columns):
    #     print(compared_str(
    #         stat_features.iloc[:, i].values[0],
    #         norma_dataset.iloc[:, i].values[0],
    #         feature.split('_')[-1]
    #     ))

    visualize_difference(stat_features, norma_dataset)
    # for i, model in enumerate(models):
    #     prediction = model.predict_proba(features)
    #
    #     predictions.append(
    #         prediction
    #     )

    return models[0].predict_proba(features)