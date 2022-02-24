from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class PredictionDataset(Dataset):
    def __init__(self, image, labels, props, half_patch_size, max_type_val):
        self.props = props
        self.image = image
        self.labels = labels
        self.half_patch_size = half_patch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.max_type_val = max_type_val

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
            maskette[maskette == prop.label] = self.max_type_val

            # L'imagette et son mask étant générés, on passe a la concaténation pour faire la prédiction du label par le CNN

            concat_image = np.zeros((imagette.shape[0], imagette.shape[1], imagette.shape[2] + 1))
            concat_image[:, :, :-1] = imagette
            concat_image[:, :, -1] = maskette
            if concat_image.shape[0] == 0 or concat_image.shape[1] == 0:
                pass
            else:
                # Normalisation de l'image
                concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

            return self.transform(concat_image.astype("float32")).to("cuda")

    def __len__(self):
        return len(self.props)
