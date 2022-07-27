import albumentations as A
from albumentations.pytorch import ToTensorV2
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

from CNN2D import CNN2D
import os
import torch
from torch import nn
import numpy as np
from skimage.io import imread, imsave

from CustomDataset import CustomDataset
from napari_svetlana.PredictionDataset import PredictionDataset
import matplotlib.pyplot as plt


def get_image_patch(image, labels, reg_props, labels_list, torch_type, case):
    """
    This function aims at contructing the tensors of the images and their labels
    """

    labels_tensor = torch.from_numpy(labels_list).type(torch_type)
    labels_tensor = nn.functional.one_hot(labels_tensor.type(torch.cuda.LongTensor))

    img_patch_list = []
    try:
        max_type_val = np.iinfo(image.dtype).max
    except ValueError:
        max_type_val = np.finfo(image.dtype).max

    for i, position in enumerate(reg_props):
        if case == "2D" or case == "multi_2D":
            xmin = (int(reg_props[i]["centroid"][0]) + (patch_size // 2) + 1) - (patch_size // 2)
            xmax = (int(reg_props[i]["centroid"][0]) + (patch_size // 2) + 1) + (patch_size // 2)
            ymin = (int(reg_props[i]["centroid"][1]) + (patch_size // 2) + 1) - (patch_size // 2)
            ymax = (int(reg_props[i]["centroid"][1]) + (patch_size // 2) + 1) + (patch_size // 2)

            imagette = image[xmin:xmax, ymin:ymax].copy()
            imagette_mask = labels[xmin:xmax, ymin:ymax].copy()

            imagette_mask[imagette_mask != reg_props[i]["label"]] = 0
            imagette_mask[imagette_mask == reg_props[i]["label"]] = max_type_val

            concat_image = np.zeros((imagette.shape[0], imagette.shape[1], image.shape[2] + 1))
            concat_image[:, :, :-1] = imagette
            concat_image[:, :, -1] = imagette_mask
            # Normalization of the image
            concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

            img_patch_list.append(concat_image)
        elif case == "multi_3D":
            xmin = (int(reg_props[i]["centroid"][1]) + (patch_size // 2) + 1) - (patch_size // 2)
            xmax = (int(reg_props[i]["centroid"][1]) + (patch_size // 2) + 1) + (patch_size // 2)
            ymin = (int(reg_props[i]["centroid"][2]) + (patch_size // 2) + 1) - (patch_size // 2)
            ymax = (int(reg_props[i]["centroid"][2]) + (patch_size // 2) + 1) + (patch_size // 2)
            zmin = (int(reg_props[i]["centroid"][0]) + (patch_size // 2) + 1) - (patch_size // 2)
            zmax = (int(reg_props[i]["centroid"][0]) + (patch_size // 2) + 1) + (patch_size // 2)

            imagette = image[:, xmin:xmax, ymin:ymax, zmin:zmax].copy()

            imagette_mask = labels[xmin:xmax, ymin:ymax, zmin:zmax].copy()

            imagette_mask[imagette_mask != reg_props[i]["label"]] = 0
            imagette_mask[imagette_mask == reg_props[i]["label"]] = max_type_val

            concat_image = np.zeros((imagette.shape[0] + 1, imagette.shape[1], imagette.shape[2],
                                     imagette.shape[3])).astype(image.dtype)

            concat_image[:-1, :, :, :] = imagette
            concat_image[-1, :, :, :] = imagette_mask

            concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

            img_patch_list.append(concat_image)

        else:
            xmin = (int(reg_props[i]["centroid"][0]) + (patch_size // 2) + 1) - (patch_size // 2)
            xmax = (int(reg_props[i]["centroid"][0]) + (patch_size // 2) + 1) + (patch_size // 2)
            ymin = (int(reg_props[i]["centroid"][1]) + (patch_size // 2) + 1) - (patch_size // 2)
            ymax = (int(reg_props[i]["centroid"][1]) + (patch_size // 2) + 1) + (patch_size // 2)
            zmin = (int(reg_props[i]["centroid"][2]) + (patch_size // 2) + 1) - (patch_size // 2)
            zmax = (int(reg_props[i]["centroid"][2]) + (patch_size // 2) + 1) + (patch_size // 2)

            imagette = image[xmin:xmax, ymin:ymax, zmin:zmax].copy()

            imagette_mask = labels[xmin:xmax, ymin:ymax, zmin:zmax].copy()

            imagette_mask[imagette_mask != reg_props[i]["label"]] = 0
            imagette_mask[imagette_mask == reg_props[i]["label"]] = max_type_val

            concat_image = np.zeros((2, imagette.shape[0], imagette.shape[1], imagette.shape[2])).astype(
                image.dtype)

            concat_image[0, :, :, :] = imagette
            concat_image[1, :, :, :] = imagette_mask

            concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

            img_patch_list.append(concat_image)

    train_data = CustomDataset(data_list=img_patch_list, labels_tensor=labels_tensor, transform=transform)
    return train_data


# Chargement du binaire
checkpoint = torch.load("/home/cazorla/Images/textures_Pierre/binary")

image_path = checkpoint["image_path"]
image = imread(image_path)
if len(image.shape) == 2:
    image = np.stack((image,) * 3, axis=-1)
labels_path = checkpoint["labels_path"]
mask = imread(labels_path)
reg_props = checkpoint["regionprops"]
labels_list = checkpoint["labels_list"]
patch_size = int(checkpoint["patch_size"])
epochs_nb = 500
training_name = "training_avg_pool"

transform = A.Compose([A.Rotate(-90, 90, p=1.0), ToTensorV2()])

# Setting of network
if image.shape[2] <= 3:
    model = CNN2D(labels_number=2, channels_nb=3 + 1).to("cuda")

elif image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
    case = "multi_2D"
    model = CNN2D(labels_number=2, channels_nb=2 + 1).to("cuda")
    image = np.transpose(image, (1, 2, 0))
"""from torchvision.models import resnet34
model = resnet34(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, max(labels_list) + 1, bias=True)
model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
"""
torch_type = torch.cuda.FloatTensor


# Setting the optimizer
LR = 0.01
torch.autograd.set_detect_anomaly(True)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
labels_list = np.array(labels_list)

# Generators
if len(mask.shape) == 2:
    pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1), (0, 0)), mode="constant")
    pad_labels = np.pad(mask, ((patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

elif len(image.shape) == 4:
    pad_image = np.pad(image, ((0, 0),
                               (patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
    pad_labels = np.pad(mask, ((patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

else:
    pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
    pad_labels = np.pad(mask, ((patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1),
                               (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

train_data = get_image_patch(pad_image, pad_labels, reg_props, labels_list, torch_type, "2D")
training_loader = DataLoader(dataset=train_data, batch_size=100, shuffle=True)

# Optimizer
model.to("cuda")
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad is True:
        params_to_update.append(param)
        print("\t", name)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Loss function
LOSS_LIST = []
weights = np.ones([max(labels_list) + 1])
weights[0] = 0
weights = torch.from_numpy(weights)
loss = nn.CrossEntropyLoss(weight=weights).type(torch_type)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
# Loop over epochs
iterations_number = epochs_nb
# folder where to save the training
save_folder = os.path.split(image_path)[0]


for epoch in range(iterations_number):
    # Training
    for local_batch, local_labels in training_loader:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        out = model(local_batch)
        total_loss = loss(out, local_labels.type(torch.cuda.FloatTensor))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        LOSS_LIST.append(total_loss.item())
        # scheduler.step()
        if (epoch + 1) % 100 == 0:
            d = {"model": model, "optimizer_state_dict": optimizer,
                 "loss": loss, "training_nb": iterations_number, "loss_list": LOSS_LIST,
                 "image_path": image_path, "labels_path": labels_path, "patch_size": patch_size}
            if training_name == "":
                model_path = os.path.join(save_folder, "training" + str(epoch + 1))
            else:
                model_path = os.path.join(save_folder, training_name + str(epoch + 1))
            if model_path.endswith(".pt") or model_path.endswith(".pth"):
                torch.save(d, model_path)
            else:
                torch.save(d, model_path + ".pth")
    if epoch % 10 == 0:
        print("Epoch ", epoch + 1)
        print(total_loss.item())


# PREDICTION:
def draw_predicted_contour(compteur, prop, imagette_contours, i, list_pred):

    imagette_contours[prop.coords[:, 0], prop.coords[:, 1]] = list_pred[i].item()
    if list_pred[i] == 1:
        compteur += 1
    return compteur

model.eval()

from skimage.measure import regionprops
props = regionprops(mask)

try:
    max = np.iinfo(image.dtype).max
except TypeError:
    max = np.finfo(image.dtype).max

compteur = 0

imagette_contours = np.zeros((image.shape[0], image.shape[1]))

data = PredictionDataset(pad_image, pad_labels, props, patch_size // 2, max)

prediction_loader = DataLoader(dataset=data, batch_size=100, shuffle=False)

global list_pred
list_pred = []
for i, local_batch in enumerate(prediction_loader):
    out = model(local_batch)
    _, index = torch.max(out, 1)
    list_pred += index

compteur = Parallel(n_jobs=-1, require="sharedmem")(
    delayed(draw_predicted_contour)(compteur, prop, imagette_contours, i, list_pred)
    for i, prop in enumerate(props))

mask1 = imread("/home/cazorla/Images/textures_Pierre/mask1.png")
mask2 = imread("/home/cazorla/Images/textures_Pierre/mask2.png")
gt = 1 * mask1 // 255 + 2 * (mask2 // 255)

from skimage.measure import regionprops, label

miss_classified = len(regionprops(label(np.abs(imagette_contours - gt))))
print("miss= ", miss_classified)

compt = 0
for i, p in enumerate(reg_props):
    x = p["coords"][0][0]
    y = p["coords"][0][1]
    lab = labels_list[i]
    if gt[x, y] != lab:
        compt += 1

print("compt=", compt)
plt.ion()
import matplotlib.pyplot as plt
plt.figure(1)
plt.imshow(imagette_contours)
plt.figure(2)
plt.imshow(np.abs(imagette_contours - gt))
plt.show()

