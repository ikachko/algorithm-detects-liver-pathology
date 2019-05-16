from os import listdir
import glob
import numpy as np
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

class IMGReader:
    def __init__(self):
        pass

    @staticmethod
    def read_directory(dir_path, format=None):
        images_name = []
        images = []
        try:
            images = [np.asarray(Image.open(img_path).convert('L'), dtype=np.uint8)
                      for img_path in glob.glob(dir_path + "*" + (("." + format) if format else ""))]
            images_name = listdir(dir_path)
        except Exception as e:
            print(e)
        return images_name, images

    @staticmethod
    def read_image(img_path):
        return np.asarray(Image.open(img_path).convert('L'), dtype=np.uint8)

def gaussian_function_2d(x, y, std):
    return (1.0 / (2 * np.pi * std ** 2)) * np.exp(-(x ** 2 + y ** 2)/(2 * (std ** 2)))

def gaussian_function_1d(x, std):
    return (1.0 / (2 * np.pi * (std ** 2))) * np.exp(-(x ** 2)/(2 * (std ** 2)))

from scipy.ndimage import gaussian_filter
from scipy.ndimage import filters

# def gaussian_filter(img, std):
#     filtered_img = np.zeros(img.shape)
#     # filtered_img = [[0] * img.shape[1]] * img.shape[0]
#
#     print(img.shape)
#     print()
#     for i in range(len(img)):
#         for j in range(len(img[0])):
#             filtered_img[i][j] = gaussian_function_1d(img[i][j], std)
#
#     print(filtered_img)
#     return filtered_img




# import matplotlib.pyplot as plt
# import os
#
#
# image = IMGReader.read_image('../data/images/png/wls/4.png')
#
# f = plt.figure()
#
# f.add_subplot(2, 2, 1)
# plt.imshow(image, cmap='gray', vmin=0, vmax=255)
# plt.title('Without Gaussian Filtering')
#
#
# sigma = 1.0
# gauss_filtered_image = gaussian_filter(image, sigma=sigma)
#
# f.add_subplot(2, 2, 2)
# plt.imshow(gauss_filtered_image, cmap='gray', vmin=0, vmax=255)
# plt.title('Gaussian Filtering, sigma={}'.format(sigma))
#
# size = 5
# median_filtered_img = filters.median_filter(image, size=size)
#
# f.add_subplot(2, 2, 3)
# plt.imshow(median_filtered_img, cmap='gray', vmin=0, vmax=255)
# plt.title('Median Filtering, size={}'.format(size))
#
# laplace_filtered_img = filters.laplace(image)
#
# f.add_subplot(2, 2, 4)
# plt.imshow(laplace_filtered_img, cmap='gray', vmin=0, vmax=255)
# plt.title('Laplace Filtering')
#
# plt.show(block=True)
