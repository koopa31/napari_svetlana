from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch
from src.napari_svetlana.PredictionDataset import max_to_1, min_max_norm
import cupy as cu
from cucim.skimage.morphology import dilation, ball
import matplotlib.pyplot as plt


class Prediction3DDataset(Dataset):
    def __init__(self, image, labels, props, half_patch_size, norm_type, device, dilation_factor):
        self.props = props
        self.image = image
        self.labels = labels
        self.half_patch_size = half_patch_size
        self.norm_type = norm_type
        self.device = device
        self.str_el = cu.asarray(ball(dilation_factor))

    def __getitem__(self, index):
        prop = self.props[index]
        if np.isnan(prop.centroid[0]) == False and np.isnan(prop.centroid[1]) == False:
            xmin = (int(prop.centroid[0]) + self.half_patch_size + 1) - self.half_patch_size
            xmax = (int(prop.centroid[0]) + self.half_patch_size + 1) + self.half_patch_size
            ymin = (int(prop.centroid[1]) + self.half_patch_size + 1) - self.half_patch_size
            ymax = (int(prop.centroid[1]) + self.half_patch_size + 1) + self.half_patch_size
            zmin = (int(prop.centroid[2]) + self.half_patch_size + 1) - self.half_patch_size
            zmax = (int(prop.centroid[2]) + self.half_patch_size + 1) + self.half_patch_size

            imagette = self.image[xmin:xmax, ymin:ymax, zmin:zmax].copy()
            maskette = self.labels[xmin:xmax, ymin:ymax, zmin:zmax].copy()

            maskette[maskette != prop.label] = 0
            maskette[maskette == prop.label] = 1

            # dilation of mask
            maskette = cu.asarray(maskette)
            maskette = cu.asnumpy(dilation(maskette, self.str_el))

            imagette *= maskette

            concat_image = np.zeros((2, imagette.shape[0], imagette.shape[1], imagette.shape[2])).astype(imagette.dtype)

            if self.norm_type == "min max normalization":
                imagette = min_max_norm(imagette)
            elif self.norm_type == "max to 1 normalization":
                imagette = max_to_1(imagette)

            concat_image[0, :, :, :] = imagette
            concat_image[1, :, :, :] = maskette

            return torch.from_numpy(concat_image.astype("float32")).to(self.device)

    def __len__(self):
        return len(self.props)
