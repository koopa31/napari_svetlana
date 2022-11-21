import pandas as pd
import cv2
import numpy as np
import os
from skimage.measure import label
from skimage.io import imread, imsave

"""
Résultats de quantification noyaux morts vivants et marqués KI-67/EDU.
"""


def generate_results_table(masks1, masks2, noyaux_positifs):

    masks1_path = sorted([os.path.join(masks1, f) for f in os.listdir(masks1) if
                           os.path.isfile(os.path.join(masks1, f)) and os.path.join(masks1, f).endswith(".png")])

    masks2_path = sorted([os.path.join(masks2, f) for f in os.listdir(masks2) if
                           os.path.isfile(os.path.join(masks2, f)) and os.path.join(masks2, f).endswith(".png")])

    edu_dapi_path = sorted([os.path.join(noyaux_positifs, f) for f in os.listdir(noyaux_positifs) if
                           os.path.isfile(os.path.join(noyaux_positifs, f)) and os.path.join(noyaux_positifs, f).endswith(".tif")])

    image_names_list = []
    wells_list = []
    fields_list = []
    validite_list = []
    cell_vivantes_list = []
    cell_mortes_list = []
    noyaux_plus_list = []

    liste = []

    for i in range(len(masks1_path)):

        image_names_list.append(os.path.splitext(os.path.split(edu_dapi_path[i])[1][11:])[0])
        wells_list.append(image_names_list[i].split("_")[-1][:3])
        fields_list.append(image_names_list[i].split("_")[-1][5])
        validite_list.append("ok")

        mask1 = imread(masks1_path[i])
        mask2 = imread(masks2_path[i])
        edu_api = imread(edu_dapi_path[i])
        edu_api[edu_api == 2] = 0
        edu_api = label(edu_api)

        cell_vivantes_list.append(len(np.unique(mask1)) - 1)
        cell_mortes_list.append(len(np.unique(mask2)) - 1)
        noyaux_plus_list.append(len(np.unique(edu_api)) - 1)

        liste.append([image_names_list[i], wells_list[i], fields_list[i], validite_list[i], cell_vivantes_list[i],
                     cell_mortes_list[i], noyaux_plus_list[i]])

    results_df = pd.DataFrame(liste, columns=["image name", "well", "field", "validity", "nuclei_nb", "dead_nuclei_nb",
                                             "positive nuclei"])

    results_df.to_excel(os.path.join(masks1[:-7], "comptage.xlsx"), engine='xlsxwriter')
