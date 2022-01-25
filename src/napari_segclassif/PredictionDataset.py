from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


class PredictionDataset(Dataset):
    def __init__(self, image, labels, props, half_patch_size):
        self.props = props
        self.image = image
        self.labels = labels
        self.half_patch_size = half_patch_size
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        prop = self.props[index]
        if np.isnan(prop.centroid[0]) == False and np.isnan(prop.centroid[1]) == False:

            imagette = self.image[int(prop.centroid[0]) - self.half_patch_size:int(prop.centroid[0]) + self.half_patch_size,
                       int(prop.centroid[1]) - self.half_patch_size:
                       int(prop.centroid[1]) + self.half_patch_size].copy()
            maskette = self.labels[int(prop.centroid[0]) - self.half_patch_size:int(prop.centroid[0]) + self.half_patch_size,
                       int(prop.centroid[1]) - self.half_patch_size:
                       int(prop.centroid[1]) + self.half_patch_size].copy()
            maskette[maskette != prop.label] = 0
            maskette[maskette == prop.label] = 255

            # L'imagette et son mask étant générés, on passe a la concaténation pour faire la prédiction du label par le CNN

            concat_image = np.zeros((imagette.shape[0], imagette.shape[1], 4))
            concat_image[:, :, :3] = imagette
            concat_image[:, :, 3] = maskette
            if concat_image.shape[0] == 0 or concat_image.shape[1] == 0:
                pass
            else:
                # Normalisation de l'image
                concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

            return self.transform(concat_image.astype("float32")).to("cuda")

    def __len__(self):
        return len(self.props)
