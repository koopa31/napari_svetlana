from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch


class Prediction3DDataset(Dataset):
    def __init__(self, image, labels, props, half_patch_size, max_type_val, device):
        self.props = props
        self.image = image
        self.labels = labels
        self.half_patch_size = half_patch_size
        self.max_type_val = max_type_val
        self.device = device

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
            maskette[maskette == prop.label] = self.max_type_val

            concat_image = np.zeros((2, imagette.shape[0], imagette.shape[1], imagette.shape[2])).astype(imagette.dtype)

            concat_image[0, :, :, :] = imagette
            concat_image[1, :, :, :] = maskette

            concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

            return torch.from_numpy(concat_image.astype("float32")).to(self.device)

    def __len__(self):
        return len(self.props)
