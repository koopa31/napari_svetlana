import os

import numpy as np
import matplotlib.pyplot as plt
import skimage.io
from cellpose import models
import pandas as pd
import argparse
import torch

from Classifier_functions import predict_batch
from calculs_masks_1et2 import calcul_masks_1et2
from mosaique_images_brutes import generate_mosa_raw_images
from mosaïque_labels import generate_mosa_labels
from mosaïque_labels_canal2 import generate_mosa_edu
from table_resultat import generate_results_table
from joblib import Parallel, delayed

"""
Le script traite des images HCD acquises au CX7 (2D). Le but étant dans un premier temps d'extraire le plus de noyaux
 possibles puis de classifier les morts.
"""

parser = argparse.ArgumentParser(description='Segmentation of cells images')
parser.add_argument('-p', '--path', metavar='N', type=str, help='Path to the folder containing the images to process')
parser.add_argument('-d', '--diameter', type=float, help='Estimation of the size of the cells', default=15.)
parser.add_argument('-c', '--channels', type=str, help='Channel to segment (gray, red, green or blue)', default="gray")

args = parser.parse_args()

# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(gpu=True, model_type='cyto2')

# list of files
# PUT PATH TO YOUR FILES HERE!

parent_folder = args.path

onlyfiles = sorted([os.path.join(parent_folder, f) for f in os.listdir(parent_folder) if
                    os.path.isdir(os.path.join(parent_folder, f))])


def run_pipeline(plate_folder, model):

    images_folder = os.path.join(plate_folder, "Images")
    masks_folder = os.path.join(plate_folder, "Masks")

    # SEGMENTATION DE TOUS LES NOYAUX AVEC CELLPOSE

    if os.path.isdir(plate_folder) is False:
        raise ValueError("This folder does not exist")

    if os.path.isdir(masks_folder) is False:
        os.mkdir(masks_folder)

    diameter = args.diameter
    if args.channels == "gray":
        channels = [[0, 0]]
    elif args.channels == "red":
        channels = [[1, 1]]
    elif args.channels == "green":
        channels = [[2, 2]]
    elif args.channels == "blue":
        channels = [[3, 3]]
    else:
        raise ValueError("The given channel name is not correct, please choose between gray, red ,green and blue")

    # Création du pandas dataframe pour stocker le fichier excel

    results_list = [[], []]

    # Liste de toutes les acquisitions
    acquisitions = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f)) and
                    os.path.join(images_folder, f).endswith(".tif") and "d0" in os.path.join(images_folder, f)])
    # On exclut le dossier résultats
    #acquisitions.sort()
    acquisitions_nb = len(acquisitions)

    # define CHANNELS to run segmentation on
    # grayscale=0, R=1, G=2, B=3
    # channels = [cytoplasm, nucleus]
    # if NUCLEUS channel does not exist, set the second channel to 0
    # channels = [[0, 0]]
    # IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
    # channels = [0,0] # IF YOU HAVE GRAYSCALE
    # channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
    # channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

    # if diameter is set to None, the size of the cells is estimated on a per image basis
    # you can set the average cell `diameter` in pixels yourself (recommended)
    # diameter can be a list or a single number for all images
    from time import time
    start = time()
    image_list = []
    for aq in acquisitions:

        print("traitement de: %s" % aq)
        image_list.append(skimage.io.imread(aq))
        results_list[0].append(aq)
    masks, flows, styles, diams = model.eval(image_list, diameter=diameter, channels=channels, cellprob_threshold=0.0,
                                                 flow_threshold=None)
        #print("nombre de cellules trouvées: %s" % masks.max())

        # Ecriture du nombre de cellules de chaque image dans une liste
    for i in range(len(masks)):
        results_list[1].append(masks[i].max())

        # Sauvegarde du masque
        result_name = os.path.join(masks_folder, "mask_" + os.path.splitext(os.path.split(acquisitions[i])[1])[0] + ".png")
        skimage.io.imsave(result_name, masks[i])

    df = pd.DataFrame(zip(*results_list))
    df.to_excel(os.path.join(masks_folder, "comptage_tous_noyaux.xlsx"), engine="xlsxwriter")
    end = time()
    print("temps de segmentation : ", end-start)

    # CLASSIFICATION DES NOYAUX MORTS

    b = torch.load(os.path.join(os.getcwd(), "training_vivant_mort.pth"))
    model_vivant_mort = b["model"].to("cuda")
    model_vivant_mort.eval()
    patch_size = b["patch_size"]
    res_folder = os.path.join(plate_folder, "Predictions_noyaux_morts")

    if os.path.isdir(res_folder) is False:
        os.mkdir(res_folder)

    image_path_list = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if
                              os.path.isfile(os.path.join(images_folder, f)) and
                              os.path.join(images_folder, f).endswith(".tif") and "d0" in os.path.join(images_folder, f)])
    mask_path_list = sorted([os.path.join(masks_folder, f) for f in os.listdir(masks_folder) if
                             os.path.isfile(os.path.join(masks_folder, f)) and
                             os.path.join(masks_folder, f).endswith(".png")])
    batch_size = 128

    predict_batch(model_vivant_mort, image_path_list, mask_path_list, patch_size, batch_size, res_folder,
                  "Config_vivant_mort.json")

    # CALCULS DES MASQUES 1 ET 2 (NOYAUX VIVANTS ET MORTS)

    calcul_masks_1et2(plate_folder)

    # CLASSIFICATION DU CANAL EDU

    b = torch.load(os.path.join(os.getcwd(), "training_edu+.pth"))
    model_edu = b["model"].to("cuda")
    model_edu.eval()
    patch_size = b["patch_size"]
    res_folder = os.path.join(plate_folder, "Predictions_noyaux_edu")
    if os.path.isdir(res_folder) is False:
        os.mkdir(res_folder)

    image_path_list = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if
                              os.path.isfile(os.path.join(images_folder, f)) and
                              os.path.join(images_folder, f).endswith(".tif") and "d1" in os.path.join(images_folder,
                                                                                                       f)])

    masks1_folder = os.path.join(plate_folder, "Masks1")
    mask_path_list = sorted([os.path.join(masks1_folder, f) for f in os.listdir(masks1_folder) if
                             os.path.isfile(os.path.join(masks1_folder, f)) and
                             os.path.join(masks1_folder, f).endswith(".png")])
    batch_size = 200

    predict_batch(model_edu, image_path_list, mask_path_list, patch_size, batch_size, res_folder, "Config_edu.json")

    # GÉNÉRATION MOSAIQUE IMAGES BRUTES
    generate_mosa_raw_images(plate_folder)

    # GÉNÉRATION MOSAIQUE OVERLAYS SEGMENTATION
    generate_mosa_labels(plate_folder)

    # GÉNÉRATION MOSAIQUE EDU +
    generate_mosa_edu(plate_folder)

    # GÉNÉRATION TABLEAU DE RÉSULTAT
    generate_results_table(os.path.join(plate_folder, "Masks1"), os.path.join(plate_folder, "Masks2"),
                           os.path.join(plate_folder, "Predictions_noyaux_edu"))

import time
start = time.time()
for plate_folder in onlyfiles:
    run_pipeline(plate_folder, model)
#Parallel(n_jobs=-1, require="sharedmem")(delayed(run_pipeline)(plate_folder, model) for plate_folder in onlyfiles)
print("ca a pris :", time.time() - start)
