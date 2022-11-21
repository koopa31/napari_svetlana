import os
import torch
import json

import numpy as np

from skimage.io import imread, imsave
from skimage.measure import regionprops
from torch.utils.data import DataLoader
from PredictionDataset import PredictionDataset
from joblib import Parallel, delayed


def draw_predicted_contour(compteur, prop, imagette_contours, i, list_pred):
    """
    Draw the mask of an object with the colour associated to its predicted class (for 2D images)
    @param compteur: counts the number of objects that belongs to class 1 (int)
    @param prop: region_property of this object
    @param imagette_contours: image of the contours
    @param i: index of the object in the list (int)
    @param list_pred: list of the labels of the classified objects
    @return:
    """

    imagette_contours[prop.coords[:, 0], prop.coords[:, 1]] = list_pred[i].item() + 1
    if list_pred[i] == 1:
        compteur += 1
    return compteur


def predict_batch(model, image_path_list, mask_path_list, patch_size, batch_size, res_folder,config_file_name):
    """
    Prediction of the class of each patch extracted from the great mask
    @param image: raw image
    @param labels: segmentation mask
    @param patch_size: size of the patches to be classified (int)
    @param batch_size: batch size for the NN (int)
    @param res_folder: path to the folder where to save the result images (str)
    @return:
    """
    with open(os.path.join(os.getcwd(), config_file_name), 'r') as f:
        config_dict = json.load(f)

    for ind in range(0, len(image_path_list)):
        image = imread(image_path_list[ind])
        labels = imread(mask_path_list[ind])
        props = regionprops(labels)

        import time
        start = time.time()
        compteur = 0
        global imagette_contours, imagette_uncertainty

        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        imagette_contours = np.zeros((image.shape[0], image.shape[1]))
        imagette_uncertainty = imagette_contours.copy()

        pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                   (patch_size // 2 + 1, patch_size // 2 + 1), (0, 0)), mode="constant")
        pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                     (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

        data = PredictionDataset(pad_image, pad_labels, props, patch_size // 2, "no normalization", "cuda", config_dict,
                                 "2D")

        prediction_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

        global list_pred
        list_pred = []
        list_proba = []
        for i, local_batch in enumerate(prediction_loader):
            with torch.no_grad():
                out = model(local_batch)
                if out.dim() == 1:
                    out = out[:, None]
                proba, index = torch.max(out, 1)
                list_pred += index
                list_proba += proba
                #yield i + 1

        print("Prediction of patches done, please wait while the result image is being generated...")
        if len(labels.shape) == 2:
            compteur = Parallel(n_jobs=-1, require="sharedmem")(
                delayed(draw_predicted_contour)(compteur, prop, imagette_contours, i, list_pred)
                for i, prop in enumerate(props))

        # Save the result automatically
        res_name = "prediction_" + os.path.split(image_path_list[ind])[1]

        imsave(os.path.join(res_folder, res_name), imagette_contours.astype(np.uint8))

        print("prediction of image " + os.path.split(image_path_list[ind])[1] + " done")
