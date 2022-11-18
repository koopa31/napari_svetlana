from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch
from .PredictionDataset import max_to_1, min_max_norm
import torch
import json
if torch.cuda.is_available() is True:
    try:
        import cupy as cu
        from cucim.skimage.morphology import dilation, ball
        cuda = True
    except ImportError:
        from skimage.morphology import ball, dilation
        cuda = False
else:
    from skimage.morphology import ball, dilation
    cuda = False


class Prediction3DDataset(Dataset):
    def __init__(self, image, labels, props, half_patch_size, norm_type, device, config_dict):
        self.props = props
        self.image = image
        self.labels = labels
        self.half_patch_size = half_patch_size
        self.norm_type = norm_type
        self.device = device
        self.config_dict = config_dict

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
            if json.loads(self.config_dict["options"]["dilation"]["dilate_mask"].lower()) is True:

                if cuda is True:
                    str_el = cu.asarray(ball(int(self.config_dict["options"]["dilation"]["str_element_size"])))
                    imagette_mask = cu.asnumpy(dilation(cu.asarray(maskette), str_el))
                else:
                    str_el = ball(int(self.config_dict["options"]["dilation"]["str_element_size"]))
                    imagette_mask = dilation(maskette, str_el)

                imagette *= imagette_mask

            concat_image = np.zeros((2, imagette.shape[0], imagette.shape[1], imagette.shape[2]))

            if self.norm_type == "min max normalization":
                imagette = min_max_norm(imagette)
            elif self.norm_type == "max to 1 normalization":
                imagette = max_to_1(imagette)

            concat_image[0, :, :, :] = imagette
            concat_image[1, :, :, :] = maskette

            return torch.from_numpy(concat_image.astype("float32")).to(self.device)

    def __len__(self):
        return len(self.props)
