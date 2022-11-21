import os.path

from skimage.io import imread, imsave
import matplotlib.pyplot as plt
import numpy as np


def calcul_masks_1et2(folder):

    masks_folder = os.path.join(folder, "Masks")
    pred_folder = os.path.join(folder, "Predictions_noyaux_morts")

    masks1_folder = os.path.join(folder, 'Masks1')
    masks2_folder = os.path.join(folder, 'Masks2')

    if os.path.isdir(masks1_folder) is False:
        os.mkdir(masks1_folder)
    if os.path.isdir(masks2_folder) is False:
        os.mkdir(masks2_folder)

    masks_list = sorted([os.path.join(masks_folder, f) for f in os.listdir(masks_folder) if
                         os.path.isfile(os.path.join(masks_folder, f)) and os.path.join(masks_folder, f).endswith(".png")])

    pred_list = sorted([os.path.join(pred_folder, f) for f in os.listdir(pred_folder) if
                        os.path.isfile(os.path.join(pred_folder, f))])

    for i in range(len(pred_list)):
        mask = imread(masks_list[i])
        pred = imread(pred_list[i])
        mask1 = mask.copy()
        mask1[pred != 1] = 0
        mask2 = mask.copy()
        mask2[pred != 2] = 0

        path1 = os.path.join(os.path.split(masks_list[i])[0] + "1", os.path.splitext(os.path.split(masks_list[i])[1])[0] + "_1" + ".png")
        path2 = os.path.join(os.path.split(masks_list[i])[0] + "2", os.path.splitext(os.path.split(masks_list[i])[1])[0] + "_2" + ".png")

        imsave(path1, mask1)
        imsave(path2, mask2)
