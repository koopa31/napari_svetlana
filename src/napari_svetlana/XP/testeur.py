import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from skimage.io import imread, imsave
from CNN2D import CNN2D
from CNN3D import CNN3D
from PredictionMulti3DDataset import PredictionMulti3DDataset
from PredictionDataset import PredictionDataset
from Prediction3DDataset import Prediction3DDataset
import os
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from time import time
from torchvision import models

from src.napari_svetlana.CustomDataset import CustomDataset


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_image_patch(image_list, mask_list, region_props_list, labels_list, torch_type, case, patch_size):
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
    if device == "cuda":
        labels_tensor = nn.functional.one_hot(labels_tensor.type(torch.cuda.LongTensor))
    else:
        labels_tensor = nn.functional.one_hot(labels_tensor.type(torch.LongTensor))

    img_patch_list = []
    try:
        max_type_val = np.iinfo(image_list[0].dtype).max
    except ValueError:
        max_type_val = np.finfo(image_list[0].dtype).max

    for i in range(0, len(image_list)):
        region_props = region_props_list[i]
        image = image_list[i]
        labels = mask_list[i]
        for i, position in enumerate(region_props):
            if case == "2D" or case == "multi_2D":
                xmin = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) - (patch_size // 2)
                xmax = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) + (patch_size // 2)
                ymin = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) - (patch_size // 2)
                ymax = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) + (patch_size // 2)

                imagette = image[xmin:xmax, ymin:ymax].copy()
                imagette_mask = labels[xmin:xmax, ymin:ymax].copy()

                imagette_mask[imagette_mask != region_props[i]["label"]] = 0
                imagette_mask[imagette_mask == region_props[i]["label"]] = 1

                concat_image = np.zeros((imagette.shape[0], imagette.shape[1], imagette.shape[2] + 1))
                #imagette = imagette / 255
                imagette = (imagette - imagette.min()) / (imagette.max() - imagette.min())
                concat_image[:, :, :-1] = imagette
                concat_image[:, :, -1] = imagette_mask
                # Image with masked of the object and inverse mask

                # concat_image[:, :, 0] = imagette[:, :, 0] * imagette_mask
                # concat_image[:, :, 1] = imagette[:, :, 0] * (1 - imagette_mask)

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
                imagette_mask[imagette_mask == region_props[i]["label"]] = max_type_val

                concat_image = np.zeros((imagette.shape[0] + 1, imagette.shape[1], imagette.shape[2],
                                         imagette.shape[3])).astype(image.dtype)

                concat_image[:-1, :, :, :] = imagette
                concat_image[-1, :, :, :] = imagette_mask

                concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

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
                imagette_mask[imagette_mask == region_props[i]["label"]] = max_type_val

                concat_image = np.zeros((2, imagette.shape[0], imagette.shape[1], imagette.shape[2])).astype(
                    image.dtype)

                concat_image[0, :, :, :] = imagette
                concat_image[1, :, :, :] = imagette_mask

                concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

                img_patch_list.append(concat_image)

    train_data = CustomDataset(data_list=img_patch_list, labels_tensor=labels_tensor, transform=transform)
    return train_data


def train(image, mask, patch_size, region_props_list, labels_list, nn_type, loss_func, lr, epochs_nb, rot, h_flip,
          v_flip, prob, batch_size, saving_ep, training_name, model=None):
    """
    Training of the classification neural network
    @param image: raw image
    @param mask: segmentation mask
    @param region_props_list: regionprops of the connected components analysis
    @param labels_list: list of the labels
    @param nn_type: dictionary containing the list of the available neural networks
    @param loss_func:Type of loss function
    @param lr: learning rate (float)
    @param epochs_nb: epochs number (int)
    @param rot: boolean to do rotation for data augmentation
    @param h_flip: boolean to do horizontal flip for data augmentation
    @param v_flip: boolean to do vertical flip for data augmentation
    @param prob: probability of data augmentation (float)
    @param batch_size: batch size for the stochastic gradient
    @param saving_ep: each how many epochs th network shall be saved
    @param training_name: name of the file containing the weights
    @param model: custom model loaded by the user (optional)
    @return:
    """

    global transform, retrain, loaded_network
    # Data augmentation
    transforms_list = []
    if rot is True:
        transforms_list.append(A.Rotate(-90, 90, p=prob))
    if h_flip is True:
        transforms_list.append(A.HorizontalFlip(p=prob))
    if v_flip is True:
        transforms_list.append(A.VerticalFlip(p=prob))
    transforms_list.append(ToTensorV2())
    transform = A.Compose(transforms_list)

    # List of available network achitectures

    nn_dict = {"ResNet18": "resnet18", "ResNet34": "resnet34", "ResNet50": "resnet50", "ResNet101": "resnet101",
               "ResNet152": "resnet152", "AlexNet": "alexnet", "DenseNet121": "densenet121",
               "DenseNet161": "densenet161", "DenseNet169": "densenet169", "DenseNet201": "densenet201",
               "lightNN_2_3": "CNN2D", "lightNN_3_5": "CNN2D", "lightNN_4_5": "CNN2D"}
    # Setting of network

    # Concatenation of all the labels lists and conversion to numpy array
    l = []
    # List where empty lists allow to remove the names of the images which have not been labelled
    labels_list_to_clean = labels_list.copy()

    for p in labels_list:
        l += p
    labels_list = np.array(l)


    if model is None:
        # 2D case
        if image.shape[2] <= 3:
            case = "2D"
            if nn_dict[nn_type] != "CNN2D":
                model = eval("models." + nn_dict[nn_type] + "(pretrained=False)")
                set_parameter_requires_grad(model, True)

                if "resnet" in nn_dict[nn_type]:
                    # The fully connected layer of the network is changed so the ouptut size is "labels_number + 1"
                    # as we have "labels_number" labels
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, max(labels_list) + 1, bias=True)
                    model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                elif "densenet" in nn_dict[nn_type]:
                    num_ftrs = model.classifier.in_features
                    model.classifier = nn.Linear(num_ftrs, max(labels_list) + 1, bias=True)
                    model.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                     bias=False)
                elif nn_dict[nn_type] == "alexnet":
                    num_ftrs = model.classifier[6].in_features
                    model.classifier[6] = nn.Linear(num_ftrs, max(labels_list) + 1, bias=True)
                    model.features[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                  bias=False)

            else:
                depth = int(nn_type.split("_")[1])
                kersize = int(nn_type.split("_")[2])

                if patch_size/(2**(depth - 1)) <=  kersize:
                    print("Patch size is too small for this network")
                model = CNN2D(max(labels_list) + 1, 4, depth, kersize)

        elif len(image.shape) == 4:
            case = "multi3D"
            image = np.transpose(image, (1, 2, 3, 0))
            mask = np.transpose(mask, (1, 2, 0))
            model = CNN3D(max(labels_list) + 1, image.shape[3] + 1)

        else:
            case = "3D"

            if case == "multi2D":
                if nn_dict[nn_type] != "CNN2D":
                    model = eval("models." + nn_dict[nn_type] + "(pretrained=False)")
                    set_parameter_requires_grad(model, True)
                    image = np.transpose(image, (1, 2, 0))

                    if "resnet" in nn_dict[nn_type]:
                        # The fully connected layer of the network is changed so the ouptut size is "labels_number + 1" as we have
                        # "labels_number" labels
                        num_ftrs = model.fc.in_features
                        model.fc = nn.Linear(num_ftrs, max(labels_list) + 1, bias=True)
                        model.conv1 = nn.Conv2d(image.shape[2] + 1, 64, kernel_size=(7, 7), stride=(2, 2),
                                                padding=(3, 3),
                                                bias=False)
                    elif "densenet" in nn_dict[nn_type]:
                        num_ftrs = model.classifier.in_features
                        model.classifier = nn.Linear(num_ftrs, max(labels_list) + 1, bias=True)
                        model.features.conv0 = nn.Conv2d(image.shape[2] + 1, 64, kernel_size=(7, 7), stride=(2, 2),
                                                         padding=(3, 3), bias=False)
                    elif nn_dict[nn_type] == "alexnet":
                        num_ftrs = model.classifier[6].in_features
                        model.classifier[6] = nn.Linear(num_ftrs, max(labels_list) + 1, bias=True)
                        model.features[0] = nn.Conv2d(image.shape[2] + 1, 64, kernel_size=(7, 7), stride=(2, 2),
                                                      padding=(3, 3), bias=False)
                else:
                    depth = int(nn_type.split("_")[1])
                    kersize = int(nn_type.split("_")[2])

                    if patch_size / (2 ** (depth - 1)) <= kersize:
                        print("Patch size is too small for this network")
                    model = CNN2D(max(labels_list) + 1, 4, depth, kersize)

            elif case == "3D":
                model = CNN3D(max(labels_list) + 1, 2)

    else:
        retrain = True
        if image.shape[2] <= 3:
            case = "2D"
        elif len(image.shape) == 4:
            case = "multi3D"
        else:
            from .CustomDialog import CustomDialog
            diag = CustomDialog()
            diag.exec()
            case = diag.get_case()

    if device == "cuda":
        torch_type = torch.cuda.FloatTensor
    else:
        torch_type = torch.FloatTensor
    losses_dict = {
        "CrossEntropy": "CrossEntropyLoss",
        "L1Smooth": "L1SmoothLoss",
        "BCE": "BceLoss",
        "Distance": "DistanceLoss",
        "L1": "L1_Loss",
        "MSE": "MseLoss"
    }

    # Computing the network's parameters number
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("NUMBER OF PARAMETERS OF THE NETWORK : ", params)

    # Setting the optimizer
    LR = lr
    torch.autograd.set_detect_anomaly(True)
    if retrain is False:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    else:
        optimizer = loaded_network["optimizer_state_dict"]

    # CUDA for PyTorch
    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    # Parameters
    # Generators
    pad_image_list = []
    pad_labels_list = []
    region_props_list_to_clean = []
    for i in range(0, len(image_path_list)):
        if len(labels_list_to_clean[i]) != 0:

            image = imread(image_path_list[i])
            mask = imread(labels_path_list[i])

            region_props_list_to_clean.append(region_props_list[i])
            if len(mask.shape) == 2:
                # Turn image into 3 channel if it is grayscale
                if len(image.shape) == 2:
                    image = np.stack((image,) * 3, axis=-1)
                pad_image_list.append(np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                                     (patch_size // 2 + 1, patch_size // 2 + 1), (0, 0)),
                                             mode="constant"))
                pad_labels_list.append(np.pad(mask, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                                     (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant"))

            elif len(image.shape) == 4:
                pad_image_list.append(np.pad(image, ((0, 0),
                                                     (patch_size // 2 + 1, patch_size // 2 + 1),
                                                     (patch_size // 2 + 1, patch_size // 2 + 1),
                                                     (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant"))
                pad_labels_list.append(np.pad(mask, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                                     (patch_size // 2 + 1, patch_size // 2 + 1),
                                                     (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant"))

            else:
                pad_image_list.append(np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                                     (patch_size // 2 + 1, patch_size // 2 + 1),
                                                     (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant"))
                pad_labels_list.append(np.pad(mask, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                                     (patch_size // 2 + 1, patch_size // 2 + 1),
                                                     (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant"))
    train_data = get_image_patch(pad_image_list, pad_labels_list, region_props_list_to_clean,
                                 labels_list, torch_type, case, patch_size)
    training_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # Optimizer
    model.to(device)
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            params_to_update.append(param)
            print("\t", name)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Loss function
    if retrain is False:
        LOSS_LIST = []
    else:
        LOSS_LIST = loaded_network["loss_list"]

    loss = eval("nn." + losses_dict[loss_func] + "().type(torch_type)")

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    # Loop over epochs
    iterations_number = epochs_nb

    # folder where to save the training
    save_folder = os.path.join(os.path.split(os.path.split(image_path_list[0])[0])[0], "Svetlana")
    if os.path.isdir(save_folder) is False:
        os.mkdir(save_folder)

    found = False
    while found is False:
        try:

            for epoch in range(iterations_number):
                print("Epoch ", epoch + 1)
                for phase in ["train", "val"]:
                    if phase == "train":
                        # Training
                        for local_batch, local_labels in training_loader:
                            # Transfer to GPU
                            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                            out = model(local_batch)
                            total_loss = loss(out, local_labels.type(torch_type))
                            optimizer.zero_grad()
                            total_loss.backward()
                            optimizer.step()
                            LOSS_LIST.append(total_loss.item())
                            print(total_loss.item())

                            # scheduler.step()
                            if (epoch + 1) % saving_ep == 0:
                                d = {"model": model, "optimizer_state_dict": optimizer,
                                     "loss": loss, "training_nb": iterations_number, "loss_list": LOSS_LIST,
                                     "image_path": image_path_list[0], "labels_path": labels_path_list[0],
                                     "patch_size": patch_size}
                                if training_name == "":
                                    model_path = os.path.join(save_folder, "training_" + str(epoch + 1))
                                else:
                                    model_path = os.path.join(save_folder, training_name + "_" + str(epoch + 1))
                                if model_path.endswith(".pt") or model_path.endswith(".pth"):
                                    torch.save(d, model_path)
                                else:
                                    torch.save(d, model_path + ".pth")

                            found = True

                    elif phase == "val":
                        pass
                epoch += 1
        except:
            if "bs" in locals():
                bs -= 1
                bs = 2 ** (np.floor(np.log(bs) / np.log(2)))
            else:
                bs = 2 ** (np.floor(np.log(batch_size) / np.log(2)))
            # We make sure bs can't be equal to zero but at least to 1
            if bs == 0:
                bs += 1
            training_loader = DataLoader(dataset=train_data, batch_size=int(bs), shuffle=True)

    """plt.plot(LOSS_LIST)
    plt.title("Training loss")
    plt.xlabel("Epochs number")
    plt.show()"""
    return model


def draw_predicted_contour(compteur, prop, imagette_contours, i, list_pred):
    """
    Draw the mask of an object with the colour associated to its predicted class (for 2D images)
    @param compteur: counts the number of objects that belongs to class 1 (int)
    @param prop: region_property of this object
    @param imagette_contours: image of the contours
    @param i: index of the object in the list (int)
    @param list_pred: list of the labels of the classified objects
    @return:
    """

    imagette_contours[prop.coords[:, 0], prop.coords[:, 1]] = list_pred[i].item() + 1
    if list_pred[i] == 1:
        compteur += 1
    return compteur


def draw_3d_prediction(compteur, prop, imagette_contours, i, list_pred):
    """
     Draw the mask of an object with the colour associated to its predicted class (for 3D images)
     @param compteur: counts the number of objects that belongs to class 1 (int)
     @param prop: region_property of this object
     @param imagette_contours: image of the contours
     @param i: index of the object in the list (int)
     @param list_pred: list of the labels of the classified objects
     @return:
     """

    imagette_contours[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = list_pred[i].item() + 1
    if list_pred[i] == 1:
        compteur += 1
    return compteur


def predict(image, labels, props, patch_size, batch_size):
    """
    Prediction of the class of each patch extracted from the great mask
    @param image: raw image
    @param labels: segmentation mask
    @param props: region_properties of the connected component analysis
    @param patch_size: size of the patches to be classified (int)
    @param batch_size: batch size for the NN (int)
    @return:
    """

    import time
    start = time.time()
    compteur = 0
    global imagette_contours

    try:
        max = np.iinfo(image.dtype).max
    except:
        max = np.finfo(image.dtype).max

    if image.shape[2] <= 3:
        case = "2D"
    elif len(image.shape) == 4:
        case = "multi3D"
    else:
        from .CustomDialog import CustomDialog
        diag = CustomDialog()
        diag.exec()
        case = diag.get_case()

    if case == "2D" or case == "multi2D":
        if case == "multi2D":
            image = np.transpose(image, (1, 2, 0))

        imagette_contours = np.zeros((image.shape[0], image.shape[1]))
        pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                   (patch_size // 2 + 1, patch_size // 2 + 1), (0, 0)), mode="constant")
        pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                     (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

        data = PredictionDataset(pad_image, pad_labels, props, patch_size // 2, max, device)

    elif case == "multi3D":
        image = np.transpose(image, (1, 2, 3, 0))
        labels = np.transpose(labels, (1, 2, 0))
        imagette_contours = np.zeros((image.shape[3], image.shape[1], image.shape[2]))
        pad_image = np.pad(image, ((0, 0),
                                   (patch_size // 2 + 1, patch_size // 2 + 1),
                                   (patch_size // 2 + 1, patch_size // 2 + 1),
                                   (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
        pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                     (patch_size // 2 + 1, patch_size // 2 + 1),
                                     (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

        data = PredictionMulti3DDataset(pad_image, pad_labels, props, patch_size // 2, max)

    else:
        imagette_contours = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                   (patch_size // 2 + 1, patch_size // 2 + 1),
                                   (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
        pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                     (patch_size // 2 + 1, patch_size // 2 + 1),
                                     (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

        data = Prediction3DDataset(pad_image, pad_labels, props, patch_size // 2, max)
    prediction_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

    global list_pred
    list_pred = []
    for i, local_batch in enumerate(prediction_loader):
        out = model(local_batch)
        _, index = torch.max(out, 1)
        list_pred += index
        i += 1

    print("Prediction of patches done, please wait while the result image is being generated...")
    if len(labels.shape) == 2:
        compteur = Parallel(n_jobs=-1, require="sharedmem")(
            delayed(draw_predicted_contour)(compteur, prop, imagette_contours, i, list_pred)
            for i, prop in enumerate(props))
    else:
        compteur = Parallel(n_jobs=-1, require="sharedmem")(
            delayed(draw_3d_prediction)(compteur, prop, imagette_contours, i, list_pred)
            for i, prop in enumerate(props))

    stop = time.time()
    print("temps de traitement", stop - start)
    print(str(np.sum(compteur)) + " objects remaining over " + str(len(props)))
    print(str(np.sum(compteur)) + " objects remaining over " + str(len(props)))

    # Save the result automatically
    res_name = "prediction_" + os.path.split(image_path_list[0])[1]
    imsave(os.path.join(res_folder, res_name), imagette_contours.astype(np.uint8))

    return imagette_contours.astype(np.uint8)


""" DEFINITION DES PARAMÈTRES """

global device
device = "cuda"

folder_path = "/mnt/86e98852-2345-4dcb-ae92-58406694998c/Documents/Test papier svetlana/Test orientation textures"
images_folder = os.path.join(folder_path, "Images")
masks_folder = os.path.join(folder_path, "Masks")
res_folder = os.path.join(folder_path, "../Predictions")
binary_file = torch.load(os.path.join(folder_path, "Svetlana", "labels"))

labels_list = binary_file["labels_list"]
patch_size = int(binary_file["patch_size"])
region_props_list = binary_file["regionprops"]
image_path_list = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder)])
labels_path_list = sorted([os.path.join(masks_folder, f) for f in os.listdir(masks_folder)])

""" TRAINING """
if device == "cuda":
    torch.cuda.synchronize("cuda")
start = time()

retrain = False
image = imread(image_path_list[0])
if len(image.shape) == 2:
    image = np.stack((image,) * 3, axis=-1)
mask = imread(labels_path_list[0])

nn_type = "lightNN_2_3"

model = train(image, mask, patch_size, region_props_list, labels_list, nn_type, loss_func="CrossEntropy", lr=0.01,
              epochs_nb=200, rot=False, h_flip=False, v_flip=False, prob=1.0, batch_size=128, saving_ep=100,
              training_name="training", model=None)

end = time()
print("training time = ", end - start)

""" PREDICTION """

from skimage.measure import regionprops, label
props = props = regionprops(mask)
model.eval()
prediction = predict(image, mask, props, patch_size, batch_size=100)

""" ACCURACY EVALUATION """

groundtruth_path = os.path.join(folder_path, "groundtruth.png")

groundtruth = imread(groundtruth_path)

labels_nb = np.max(imread(os.path.join(folder_path, "mask_cp_masks.tif")))

plt.figure(1)
plt.imshow(groundtruth)
plt.title("gt")
plt.figure(2)
plt.imshow(prediction)
plt.title("prediction")

diff = np.abs(groundtruth.astype(np.int) - prediction.astype(np.int))

plt.figure(3)
plt.imshow(diff)
plt.title("diff")


regprops_gt = regionprops(groundtruth)
regprops_result = regionprops(label(diff))

accuracy = 1 - (len(regprops_result) / labels_nb)

print("accuracy = ", accuracy)
plt.show()
