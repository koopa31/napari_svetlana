import albumentations as A
from albumentations.pytorch import ToTensorV2
from joblib import Parallel, delayed
from torch.utils.data import DataLoader

from CNN2D import CNN2D
from CNN3D import CNN3D
import os
import torch
from torch import nn
import numpy as np
from skimage.io import imread, imsave

from src.napari_svetlana.CustomDataset import CustomDataset
from src.napari_svetlana.Prediction3DDataset import Prediction3DDataset
import matplotlib.pyplot as plt


"""
SCRIPT TO FIND THE OPTIMAL PATCH SIZE AND THE OPTIMAL NORMALIZATION FOR A 3D SAMPLE
"""


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


def get_image_patch(image, labels, region_props, labels_list, torch_type, case, norm_type):
    """
    This function aims at contructing the tensors of the images and their labels
    @param image: Raw image
    @param labels: Segmentation mask
    @param region_props_list: regionprops of the connected components analysis
    @param labels_list: List of the labels
    @param torch_type: type of the tensors
    @param case: indicates if the image is 2D, 3D or multichannel
    @return:
    """

    labels_tensor = torch.from_numpy(labels_list).type(torch_type)
    labels_tensor = nn.functional.one_hot(labels_tensor.type(torch.cuda.LongTensor))

    img_patch_list = []

    for i, position in enumerate(region_props):
        if case == "2D" or case == "multi2D":
            xmin = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) - (patch_size // 2)
            xmax = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) + (patch_size // 2)
            ymin = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) - (patch_size // 2)
            ymax = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) + (patch_size // 2)

            imagette = image[xmin:xmax, ymin:ymax].copy()
            imagette_mask = labels[xmin:xmax, ymin:ymax].copy()

            imagette_mask[imagette_mask != region_props[i]["label"]] = 0
            imagette_mask[imagette_mask == region_props[i]["label"]] = 1

            concat_image = np.zeros((imagette.shape[0], imagette.shape[1], imagette.shape[2] + 1))
            # imagette = imagette / 255

            if norm_type == "min max normalization":
                imagette = min_max_norm(imagette)
            elif norm_type == "max to 1 normalization":
                imagette = max_to_1(imagette)

            concat_image[:, :, :-1] = imagette
            concat_image[:, :, -1] = imagette_mask
            # Image with masked of the object and inverse mask

            # concat_image[:, :, 0] = np.mean(imagette[:, :, 0] * imagette_mask) #torch.ones_like(imagette[:, :, 0])
            # concat_image[:, :, 0] *= torch.mean(imagette[:, :, 0] * imagette_mask)

            # concat_image[:, :, 1] = (imagette[:, :, 0] * (1 - imagette_mask))

            img_patch_list.append(concat_image)
        elif case == "multi3D":
            xmin = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) - (patch_size // 2)
            xmax = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) + (patch_size // 2)
            ymin = (int(region_props[i]["centroid"][2]) + (patch_size // 2) + 1) - (patch_size // 2)
            ymax = (int(region_props[i]["centroid"][2]) + (patch_size // 2) + 1) + (patch_size // 2)
            zmin = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) - (patch_size // 2)
            zmax = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) + (patch_size // 2)

            imagette = image[:, zmin:zmax, xmin:xmax, ymin:ymax].copy()

            imagette_mask = labels[zmin:zmax, xmin:xmax, ymin:ymax].copy()

            imagette_mask[imagette_mask != region_props[i]["label"]] = 0
            imagette_mask[imagette_mask == region_props[i]["label"]] = 1

            concat_image = np.zeros((imagette.shape[0] + 1, imagette.shape[1], imagette.shape[2],
                                     imagette.shape[3])).astype(image.dtype)

            # normalization
            if norm_type == "min max normalization":
                imagette = min_max_norm(imagette)
            elif norm_type == "max to 1 normalization":
                imagette = max_to_1(imagette)

            concat_image[:-1, :, :, :] = imagette
            concat_image[-1, :, :, :] = imagette_mask

            img_patch_list.append(concat_image)

        else:
            xmin = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) - (patch_size // 2)
            xmax = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) + (patch_size // 2)
            ymin = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) - (patch_size // 2)
            ymax = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) + (patch_size // 2)
            zmin = (int(region_props[i]["centroid"][2]) + (patch_size // 2) + 1) - (patch_size // 2)
            zmax = (int(region_props[i]["centroid"][2]) + (patch_size // 2) + 1) + (patch_size // 2)

            imagette = image[xmin:xmax, ymin:ymax, zmin:zmax].copy()

            imagette_mask = labels[xmin:xmax, ymin:ymax, zmin:zmax].copy()

            imagette_mask[imagette_mask != region_props[i]["label"]] = 0
            imagette_mask[imagette_mask == region_props[i]["label"]] = 1

            concat_image = np.zeros((2, imagette.shape[0], imagette.shape[1], imagette.shape[2])).astype(
                image.dtype)

            # normalization
            if norm_type == "min max normalization":
                imagette = min_max_norm(imagette)
            elif norm_type == "max to 1 normalization":
                imagette = max_to_1(imagette)

            concat_image[0, :, :, :] = imagette
            concat_image[1, :, :, :] = imagette_mask

            img_patch_list.append(concat_image)

    train_data = CustomDataset(data_list=img_patch_list, labels_tensor=labels_tensor, transform=transform)
    return train_data


# Chargement du binaire
checkpoint = torch.load("/home/cazorla/Images/Test papier svetlana/tube neural 3d/Svetlana/labels")
image_path = checkpoint["image_path"]

for norm_type in ["min max normalization", "max to 1 normalization", "no normalization"]:
    for patch_size in [45, 50, 55, 60, 65, 70, 75, 79]:

        image = imread(image_path[0])
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        labels_path = checkpoint["labels_path"][0]
        mask = imread(labels_path)
        reg_props = checkpoint["regionprops"][0]
        labels_list = checkpoint["labels_list"][0]
        #patch_size = int(checkpoint["patch_size"])
        if patch_size <= 60:
            epochs_nb = 600
        else:
            epochs_nb = 300
        print(epochs_nb)
        training_name = norm_type + "_patchsize_" + str(patch_size) + "_"

        transform = A.Compose([ToTensorV2()])
        #norm_type = "no normalization"

        # Setting of network
        model = CNN3D(labels_number=2, channels_nb=2)

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

        train_data = get_image_patch(pad_image, pad_labels, reg_props, labels_list, torch_type, "3D", norm_type)
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

        loss = nn.CrossEntropyLoss().type(torch_type)

        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        # Loop over epochs
        iterations_number = epochs_nb
        # folder where to save the training
        save_folder = os.path.join(os.path.split(os.path.split(image_path[0])[0])[0], 'Test_patch_size')

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
                         "image_path": image_path, "labels_path": labels_path,
                         "patch_size": patch_size, "norm_type": norm_type}
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

            imagette_contours[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = list_pred[i].item() + 1
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

        imagette_contours = np.zeros((image.shape[0], image.shape[1], image.shape[2]))

        data = Prediction3DDataset(pad_image, pad_labels, props, patch_size // 2, norm_type, "cuda")

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

        print(imagette_contours.shape, imagette_contours.dtype)
        imsave(os.path.join(save_folder, training_name + ".tif"), imagette_contours.astype("uint8"))




