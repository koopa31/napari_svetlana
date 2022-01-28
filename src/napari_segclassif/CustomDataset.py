import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_list, labels_tensor, transform=None):
        self.data_list = data_list
        self.labels_tensor = labels_tensor
        self.transform = transform

    def __getitem__(self, index):
        if self.transform is not None:
            if len(self.data_list[index].shape) == 4:
                transformed = torch.from_numpy(self.data_list[index].astype("float32"))
                return transformed, self.labels_tensor[index]
            else:
                transformed = self.transform(image=self.data_list[index].astype("float32"))
                return transformed["image"], self.labels_tensor[index]
        else:
            return self.data_list[index].astype("float32"), self.labels_tensor[index]

    def __len__(self):
        return len(self.data_list)
