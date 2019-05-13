import os
import nrrd
import numpy as np
import pandas as pd

from img_reader import IMGReader
from scipy.ndimage import filters
from radiomics import featureextractor


IMG_PWD = os.getcwd() + '/data/images/png'


NORMA_DIR = IMG_PWD + '/norm/'

norm_img_names, norm_img = IMGReader.read_directory(NORMA_DIR)

# Discholia
dsh_img_names, dsh_img = IMGReader.read_directory(IMG_PWD + "/dsh/")

# Hepatitis B
hpb_img_names, hpb_img = IMGReader.read_directory(IMG_PWD + "/hpb/")

# Hepatitis C
hpc_img_names, hpc_img = IMGReader.read_directory(IMG_PWD + "/hpc/")

# Wilson's disease
wls_img_names, wls_img = IMGReader.read_directory(IMG_PWD + "/wls/")

# Autoimmune hepatitis
auh_img_names, auh_img = IMGReader.read_directory(IMG_PWD + "/auh/")


# Create target directory if don't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        print("Directory ", directory, " Created ")
    # else:
    #     print("Directory ", directory, " already exists")


def gaussian_filter_dump(images, std):
    return np.asarray(list(map(lambda x: filters.gaussian_filter(x, std), images)))


def median_filter_convert(images, window):
    return np.asarray(list(map(lambda x: filters.median_filter(x, window), images)))


sl = "/"
folderNames = [
    "norm",
    "auh",
    "dsh",
    "hpb",
    "hpc",
    "wls"
]

images = [
    norm_img,
    auh_img,
    dsh_img,
    hpb_img,
    hpc_img,
    wls_img
]

gauss_filter_imgs = [
    gaussian_filter_dump(norm_img, 1.0),
    gaussian_filter_dump(dsh_img, 1.0),
    gaussian_filter_dump(auh_img, 1.0),
    gaussian_filter_dump(hpb_img, 1.0),
    gaussian_filter_dump(hpc_img, 1.0),
    gaussian_filter_dump(wls_img, 1.0)
]

median_filer_imgs = [
    median_filter_convert(norm_img, 5),
    median_filter_convert(dsh_img, 5),
    median_filter_convert(auh_img, 5),
    median_filter_convert(hpb_img, 5),
    median_filter_convert(hpc_img, 5),
    median_filter_convert(wls_img, 5)
]

# Dump filtered images to .nrrd

# data_array = list()
#
# for i in range(len(folderNames)):
#     size = len(gauss_filter_imgs[i])
#     for j in range(size):
#         image = gauss_filter_imgs[i][j]
#
#         # Add 1 additional axis for future Radiomics processing
#         image = image[..., np.newaxis]
#         label = np.ones(shape=image.shape)
#
#         # Declare destination of the data
#         folder = os.getcwd() + "/data/images/nrrd/gauss_filter/" + folderNames[i]
#
#         create_directory(folder)
#
#         name_image = folderNames[i] + "_image_" + str(j) + ".nrrd"
#         name_label = folderNames[i] + "_label_" + str(j) + ".nrrd"
#         image_path_to = folder + sl + name_image
#         label_path_to = folder + sl + name_label
#
#         # Save the image as NRRD
#         nrrd.write(image_path_to, image)
#         nrrd.write(label_path_to, label)
#
#         # Instantiate the extractor
#         extractor = featureextractor.RadiomicsFeatureExtractor()
#         extractor.disableAllFeatures()
#         extractor.enableFeatureClassByName('firstorder')
#         extractor.enableFeatureClassByName('glcm')
#         extractor.enableFeatureClassByName('glrlm')
#         extractor.enableFeatureClassByName('ngtdm')
#         extractor.enableFeatureClassByName('gldm')
#
#         result = extractor.execute(image_path_to, label_path_to)
#         result['data_source'] = folderNames[i] + str(j)
#         result['diagnosis_code'] = i
#         # for key, value in result.items():
#         #    print(key, ":", value)
#         data_array.append(result)
#
# df = pd.DataFrame(data_array)
# df.to_csv(os.getcwd() + '/data/datasets/' + 'pyradiomics_features_gaussian_filter.csv', sep=';')

data_array = list()

for i in range(len(folderNames)):
    size = len(median_filer_imgs[i])
    for j in range(size):
        image = median_filer_imgs[i][j]

        # Add 1 additional axis for future Radiomics processing
        image = image[..., np.newaxis]
        label = np.ones(shape=image.shape)

        # Declare destination of the data
        folder = os.getcwd() + "/data/images/nrrd/median_filter/" + folderNames[i]

        create_directory(folder)

        name_image = folderNames[i] + "_image_" + str(j) + ".nrrd"
        name_label = folderNames[i] + "_label_" + str(j) + ".nrrd"
        image_path_to = folder + sl + name_image
        label_path_to = folder + sl + name_label

        # Save the image as NRRD
        nrrd.write(image_path_to, image)
        nrrd.write(label_path_to, label)

        # Instantiate the extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('ngtdm')
        extractor.enableFeatureClassByName('gldm')

        result = extractor.execute(image_path_to, label_path_to)
        result['data_source'] = folderNames[i] + str(j)
        result['diagnosis_code'] = i
        # for key, value in result.items():
        #    print(key, ":", value)
        data_array.append(result)

df = pd.DataFrame(data_array)
df.to_csv(os.getcwd() + '/data/datasets/' + 'pyradiomics_features_median_filter.csv', sep=';')




# Read .nrrd images and dump pyradiomics features to .csv



