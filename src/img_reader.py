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
