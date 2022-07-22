import matplotlib.pyplot as plt
import torch
from skimage.io import imread, imsave
import os
import numpy as np

binary = torch.load("/home/cazorla/Images/Test papier svetlana/tube neural 3d/Svetlana/labels")

image = imread(binary["image_path"][0])
mask = imread(binary["labels_path"][0])
region_props = binary["regionprops"]
patch_size = int(binary["patch_size"])
labs = binary["labels_list"][0]

res_folder = "/home/cazorla/Bureau/patches"
lab1_folder = os.path.join(res_folder, "label1")
lab2_folder = os.path.join(res_folder, "label2")

image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                                         (patch_size // 2 + 1, patch_size // 2 + 1),
                                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
mask = np.pad(mask, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                     (patch_size // 2 + 1, patch_size // 2 + 1),
                                     (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

for i, position in enumerate(region_props[0]):
    xmin = int(region_props[0][i]["centroid"][0]) + (patch_size // 2) + 1 - (patch_size // 2)
    xmax = int(region_props[0][i]["centroid"][0]) + (patch_size // 2) + 1 + (patch_size // 2)
    ymin = int(region_props[0][i]["centroid"][1]) + (patch_size // 2) + 1 - (patch_size // 2)
    ymax = int(region_props[0][i]["centroid"][1]) + (patch_size // 2) + 1 + (patch_size // 2)
    zmin = int(region_props[0][i]["centroid"][2]) + (patch_size // 2) + 1 - (patch_size // 2)
    zmax = int(region_props[0][i]["centroid"][2]) + (patch_size // 2) + 1 + (patch_size // 2)

    imagette = image[xmin:xmax, ymin:ymax, zmin:zmax].copy()
    maskette = mask[xmin:xmax, ymin:ymax, zmin:zmax].copy()

    maskette[maskette != int(region_props[0][i]["label"])] = 0
    maskette[maskette != 0] = 255

    if labs[i] == 0:
        imsave(os.path.join(res_folder, lab1_folder, "patch_" + str(i + 1)+ ".tif"), imagette)
        imsave(os.path.join(res_folder, lab1_folder, "mask_" + str(i + 1)+ ".tif"), maskette)

    elif labs[i] == 1:
        imsave(os.path.join(res_folder, lab2_folder, "patch_" + str(i + 1) + ".tif"), imagette)
        imsave(os.path.join(res_folder, lab2_folder, "mask_" + str(i + 1) + ".tif"), maskette)