import time

import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.measure import regionprops
import torch
from torch.utils.data import DataLoader
from skimage.io import imread, imsave

from src.napari_svetlana.Prediction3DDataset import Prediction3DDataset
from joblib import Parallel, delayed
from natsort import natsorted
import pandas as pd


def draw_3d_prediction(compteur, prop, imagette_contours, i, list_pred):

    if list_pred[i] == 1:
        imagette_contours[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = list_pred[i].item()
        compteur += 1
    return compteur


image_folder = "/mnt/86e98852-2345-4dcb-ae92-58406694998c/Documents/Classif_KI67_220516/Images"
labels_folder = "/mnt/86e98852-2345-4dcb-ae92-58406694998c/Documents/Classif_KI67_220516/Masks"
result_folder = "/mnt/86e98852-2345-4dcb-ae92-58406694998c/Documents/Classif_KI67_220516/Results_700_aug"

images = natsorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])
labels = natsorted([os.path.join(labels_folder, f) for f in os.listdir(labels_folder) if os.path.isfile(os.path.join(labels_folder, f))])

b = torch.load("/mnt/86e98852-2345-4dcb-ae92-58406694998c/Documents/Classif_KI67_220516/training aug rot flip/training700.pth")
patch_size = b["patch_size"]
model = b["model"]
model.eval()

results_list = [[], []]

from time import time
for i in range(0, len(images)):
    start = time()
    compteur = 0

    image = imread(images[i])
    """if image.shape[0] == 2:
        image = image[1, :, :, :]
    else:
        image = image[:, 1, :, :]"""
    label = imread(labels[i])

    props = regionprops(label)

    try:
        max = np.iinfo(image.dtype).max
    except:
        max = np.finfo(image.dtype).max

    imagette_contours = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
    pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
    pad_labels = np.pad(label, ((patch_size // 2 + 1, patch_size // 2 + 1),
                        (patch_size // 2 + 1, patch_size // 2 + 1),
                        (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

    data = Prediction3DDataset(pad_image, pad_labels, props, patch_size // 2, max)

    prediction_loader = DataLoader(dataset=data, batch_size=100, shuffle=False)

    global list_pred
    list_pred = []
    for local_batch in prediction_loader:
        out = model(local_batch)
        _, index = torch.max(out, 1)
        list_pred += index

    print("Prediction of patches done, please wait while the result image is being generated...")

    compteur = Parallel(n_jobs=-1, require="sharedmem")(
        delayed(draw_3d_prediction)(compteur, prop, imagette_contours, i, list_pred)
        for i, prop in enumerate(props))

    imsave(os.path.join(result_folder, "classif_" + os.path.split(images[i])[1]), imagette_contours)
    print("rate = ", np.sum(compteur) / len(props))
    results_list[0].append(os.path.split(images[i])[1])
    results_list[1].append(np.sum(compteur) / len(props))
    print("temps de traitement", time() - start)

df = pd.DataFrame(zip(*results_list))
df.to_excel(os.path.join(result_folder, "tableur.xlsx"), engine="xlsxwriter")
