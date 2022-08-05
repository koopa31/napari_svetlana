from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import torch
import json
if torch.cuda.is_available() is True:
    try:
        import cupy as cu
        from cucim.skimage.morphology import dilation, disk

        cuda = True
    except ImportError:
        from skimage.morphology import dilation, disk
        cuda = False
else:
    from skimage.morphology import dilation, disk
    cuda = False


def min_max_norm(im):
    """
    min max normalization of an image
    @param im: Numpy array
    @return: Normalized Numpy array
    """
    im = (im - im.min()) / (im.max() - im.min())
    return im


def max_to_1(im):
    """
    Standardization of a function
    @param im: Numpy array
    @return: Standardized Numpy array
    """
    im = im / im.max()
    return im


class PredictionDataset(Dataset):
    def __init__(self, image, labels, props, half_patch_size, norm_type, device, config_dict, case):
        self.props = props
        self.image = image
        self.labels = labels
        self.half_patch_size = half_patch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.norm_type = norm_type
        self.device = device
        self.config_dict = config_dict
        self.case = case

    def __getitem__(self, index):
        prop = self.props[index]
        if np.isnan(prop.centroid[0]) == False and np.isnan(prop.centroid[1]) == False:
            xmin = (int(prop.centroid[0]) + self.half_patch_size + 1) - self.half_patch_size
            xmax = (int(prop.centroid[0]) + self.half_patch_size + 1) + self.half_patch_size
            ymin = (int(prop.centroid[1]) + self.half_patch_size + 1) - self.half_patch_size
            ymax = (int(prop.centroid[1]) + self.half_patch_size + 1) + self.half_patch_size

            imagette = self.image[xmin:xmax, ymin:ymax].copy()
            maskette = self.labels[xmin:xmax, ymin:ymax].copy()

            maskette[maskette != prop.label] = 0
            maskette[maskette == prop.label] = 1

            if json.loads(self.config_dict["options"]["dilation"]["dilate_mask"].lower()) is True:

                if cuda is True:
                    str_el = cu.asarray(disk(int(self.config_dict["options"]["dilation"]["str_element_size"])))
                    maskette = cu.asnumpy(dilation(cu.asarray(maskette), str_el))
                else:
                    str_el = disk(int(self.config_dict["options"]["dilation"]["str_element_size"]))
                    maskette = dilation(maskette, str_el)

                if self.case == "2D":
                    imagette *= maskette[:, :, None]
                else:
                    imagette *= maskette[None, :, :]

            # L'imagette et son mask étant générés, on passe a la concaténation pour faire la prédiction du label par
            # le CNN

            concat_image = np.zeros((imagette.shape[0], imagette.shape[1], imagette.shape[2] + 1))
            # imagette = imagette / 255

            if self.norm_type == "min max normalization":
                imagette = min_max_norm(imagette)
            elif self.norm_type == "max to 1 normalization":
                imagette = max_to_1(imagette)

            concat_image[:, :, :-1] = imagette
            concat_image[:, :, -1] = maskette

            # Image with masked of the object and inverse mask
            #concat_image[:, :, 0] = imagette[:, :, 0] * maskette
            #concat_image[:, :, 1] = imagette[:, :, 0] * (1 - maskette)
            if concat_image.shape[0] == 0 or concat_image.shape[1] == 0:
                pass

            return self.transform(concat_image.astype("float32")).to(self.device)

    def __len__(self):
        return len(self.props)
