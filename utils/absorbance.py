import numpy as np


def absorbance(reflectance):
    return np.log10(1 / reflectance)
