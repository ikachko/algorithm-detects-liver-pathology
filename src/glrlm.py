import numpy as np


class GLRLM:
    """
    GLRLM - Gray level run length matrix
    """

    def __init__(self, image):
        self.image = image
        self.h, self.w = np.shape(image)
        self.gray_level = 255
        self.glrlm = None

    def glrlm_0(self):
        """Vertical run"""

        glrlm = np.zeros([self.gray_level, self.w], dtype=int)

        for i in range(self.h):
            count = 0
            for j in range(self.w):
                if j < self.w - 1 and self.image[i][j] == self.image[i][j + 1]:
                    count += 1
                else:
                    glrlm[self.image[i][j]][count] += 1
                    count = 0
        return glrlm

    def LGRE(self, glrlm):
        res = 0
        s = glrlm.shape[1]

        for i in range(glrlm.shape[0]):
            for j in range(glrlm.shape[1]):
                res += (glrlm[i][j] / s) / (i * i) if i != 0 else 0

        return res

    def HGRE(self, glrlm):
        res = 0
        s = glrlm.shape[1]

        for i in range(glrlm.shape[0]):
            for j in range(glrlm.shape[1]):
                res += (glrlm[i][j] * (i * i)) / s

        return res

    def GLNU(self, glrlm):
        res = 0

        for i in range(glrlm.shape[0]):
            for j in range(glrlm.shape[1]):
                res += glrlm[i][j]**2

        return res

#
# image = np.asarray([
#     [0, 1, 2, 3],
#     [0, 2, 3, 3],
#     [2, 1, 1, 1],
#     [3, 0, 3, 0]
# ])
#
# gl = GLRLM(image)
#
# print(gl.glrlm_0())
# res = gl.glrlm_0()
#
# print("LGRE: ", gl.LGRE(res))
# print("HGRE: ", gl.HGRE(res))
# print("GLNU: ", gl.GLNU(res))

from glcm import GLCM
from img_reader import IMGReader

import numpy as np
import pandas as pd

HOME_PROJECT_PWD = "/Users/ilyakachko/algorithm-detects-liver-pathology"

UNIT_PROJECT_PWD = "/Volumes/Storage/goinfre/ikachko/algorithm-detects-liver-pathology"

# NORMA_DIR = HOME_PROJECT_PWD + "/norma_png/"
# PATHOLOGY_DIR = HOME_PROJECT_PWD + "/pathology_png/"

NORMA_DIR = UNIT_PROJECT_PWD + "/images/norma_png/"
PATHOLOGY_DIR = UNIT_PROJECT_PWD + "/images/"

norma_imgs_names, norma_imgs = IMGReader.read_directory(NORMA_DIR)
pathology_img_names, pathology_imgs = IMGReader.read_directory(PATHOLOGY_DIR)

# Discholia
dsh_img_names, dsh_img = IMGReader.read_directory(PATHOLOGY_DIR + "/dsh/")

# Hepatitis B
gpb_img_names, gpb_img = IMGReader.read_directory(PATHOLOGY_DIR + "/gpb/")

# Hepatitis C
gpc_img_names, gpc_img = IMGReader.read_directory(PATHOLOGY_DIR + "/gpc/")

# Wilson's disease
vls_img_names, vls_img = IMGReader.read_directory(PATHOLOGY_DIR + "/vls/")

# Autoimmune hepatitis
auh_img_names, auh_img = IMGReader.read_directory(PATHOLOGY_DIR + "/vls/")

print("{} norma images.".format(len(norma_imgs)))
print("{} patho images.".format(len(pathology_imgs)))

df = pd.DataFrame(columns=['LGRE', 'HGRE', 'GLNU', 'isPatho'])

i = 0
for n_img in norma_imgs:
    g = GLRLM(n_img)

    g_0 = g.glrlm_0()
    df.loc[i] = [
        g.LGRE(g_0),
        g.HGRE(g_0),
        g.GLNU(g_0),
        0
    ]
    i += 1

for img in dsh_img:
    g = GLRLM(img)

    g_0 = g.glrlm_0()
    df.loc[i] = [
        g.LGRE(g_0),
        g.HGRE(g_0),
        g.GLNU(g_0),
        1
    ]
    i += 1

for img in gpb_img:
    g = GLRLM(img)

    g_0 = g.glrlm_0()
    df.loc[i] = [
        g.LGRE(g_0),
        g.HGRE(g_0),
        g.GLNU(g_0),
        2
    ]
    i += 1

for img in gpc_img:
    g = GLRLM(img)

    g_0 = g.glrlm_0()
    df.loc[i] = [
        g.LGRE(g_0),
        g.HGRE(g_0),
        g.GLNU(g_0),
        3
    ]
    i += 1

for img in vls_img:
    g = GLRLM(img)

    g_0 = g.glrlm_0()
    df.loc[i] = [
        g.LGRE(g_0),
        g.HGRE(g_0),
        g.GLNU(g_0),
        4
    ]
    i += 1

for img in auh_img:
    g = GLRLM(img)

    g_0 = g.glrlm_0()
    df.loc[i] = [
        g.LGRE(g_0),
        g.HGRE(g_0),
        g.GLNU(g_0),
        5
    ]
    i += 1

df.to_csv('./datasets/glrlm_0_sep_diseases.csv')
print(df.head())
