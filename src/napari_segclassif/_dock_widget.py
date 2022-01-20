"""
SegClassif dock widget module
"""
import functools
import os
import pickle
import random
from typing import Any

import cv2
import matplotlib.pyplot as plt
from napari_plugin_engine import napari_hook_implementation
from napari.utils.notifications import show_info, notification_manager

import time
import numpy as np

from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui

from skimage.measure import regionprops
from skimage.morphology import ball, erosion
from skimage.io import imread

from CustomDataset import CustomDataset
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torchvision import transforms

from superqt import ensure_main_thread
from qtpy.QtWidgets import QFileDialog

@ensure_main_thread
def show_info(message: str):
    notification_manager.receive_info(message)


# @thread_worker
def read_logging(log_file, logwindow):
    with open(log_file, 'r') as thefile:
        # thefile.seek(0,2) # Go to the end of the file
        while True:
            line = thefile.readline()
            if not line:
                time.sleep(0.01)  # Sleep briefly
                continue
            else:
                logwindow.cursor.movePosition(logwindow.cursor.End)
                logwindow.cursor.insertText(line)
                yield line


labels_number = [('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6)]
networks_list = ["ResNet18", "GoogleNet", "DenseNet"]
losses_list = ["CrossEntropy", "L1Smooth", "BCE", "Distance", "L1", "MSE"]

counter = 0
labels_list = []
# logo = os.path.join(__file__, 'logo/logo_small.png')


def Annotation():
    from napari.qt.threading import thread_worker

    @Viewer.bind_key('1')
    def set_label_1(viewer):
        global counter

        if counter < len(patch[2]) - 1:
            labels_list.append(1)
            counter += 1
            if patch[2][counter].shape[2] > 3:
                # if the image is 3D, we switch to 3D view and to display the overlay of patch and mask patch
                viewer.layers.pop()
                viewer.layers.pop()
                viewer.add_labels(patch[1][counter].astype("int"))
                viewer.add_image(patch[2][counter])
            else:
                # 2D case
                viewer.layers.pop()
                viewer.add_image(patch[2][counter])
            print("label 1", labels_list)
            viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
        elif counter == len(patch[2]) - 1:
            labels_list.append(1)
            if patch[2][counter].shape[2] > 3:
                viewer.layers.pop()
                viewer.layers.pop()
            else:
                viewer.layers.pop()
            counter += 1
            from skimage.io import imread
            viewer.add_image(imread("https://bitbucket.org/koopa31/napari_package_images/raw/"
                                    "a9fda1dd3361880162474cf0b30119b1e188f53c/image_finish.png"))
            print("annotation over", labels_list)
            viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
            show_info("Annotation over, press 1 to save the result")
        else:
            # Saving of the annotation result in a binary file
            path = QFileDialog.getSaveFileName(None, 'Save File', options=QFileDialog.DontUseNativeDialog)[0]
            res_dict = {"image_path": image_path, "labels_path": labels_path, "regionprops": mini_props_list,
                        "labels_list": labels_list, "patch_size": annotation_widget.patch_size.value}
            """
            with open(path, "wb") as handle:
                pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            """
            torch.save(res_dict, path)

    @Viewer.bind_key('2')
    def set_label_2(viewer):
        global counter
        if counter < len(patch[2]) - 1:
            labels_list.append(2)
            counter += 1
            if patch[2][counter].shape[2] > 3:
                # if the image is 3D, we switch to 3D view and to display the overlay of patch and mask patch
                viewer.layers.pop()
                viewer.layers.pop()
                viewer.add_labels(patch[1][counter].astype("int"))
                viewer.add_image(patch[2][counter])
            else:
                # 2D case
                viewer.layers.pop()
                viewer.add_image(patch[2][counter])
            print("label 2", labels_list)
            viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
        elif counter == len(patch[2]) - 1:
            labels_list.append(2)
            if patch[2][counter].shape[2] > 3:
                viewer.layers.pop()
                viewer.layers.pop()
            else:
                viewer.layers.pop()
            counter += 1
            from skimage.io import imread
            viewer.add_image(imread("https://bitbucket.org/koopa31/napari_package_images/raw/"
                                    "a9fda1dd3361880162474cf0b30119b1e188f53c/image_finish.png"))
            print("annotation over", labels_list)
            viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
            show_info("Annotation over, press 1 to save the result")
        else:
            pass

    @Viewer.bind_key('3')
    def set_label_3(viewer):
        global counter
        if (int(annotation_widget.labels_nb.value) < 3) is False:
            if counter < len(patch[2]) - 1:
                labels_list.append(3)
                counter += 1
                if patch[2][counter].shape[2] > 3:
                    # if the image is 3D, we switch to 3D view and to display the overlay of patch and mask patch
                    viewer.layers.pop()
                    viewer.layers.pop()
                    viewer.add_labels(patch[1][counter].astype("int"))
                    viewer.add_image(patch[2][counter])
                else:
                    # 2D case
                    viewer.layers.pop()
                    viewer.add_image(patch[2][counter])
                print("label 2", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
            elif counter == len(patch[2]) - 1:
                labels_list.append(3)
                if patch[2][counter].shape[2] > 3:
                    viewer.layers.pop()
                    viewer.layers.pop()
                else:
                    viewer.layers.pop()
                counter += 1
                from skimage.io import imread
                viewer.add_image(imread("https://bitbucket.org/koopa31/napari_package_images/raw/"
                                        "a9fda1dd3361880162474cf0b30119b1e188f53c/image_finish.png"))
                print("annotation over", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
                show_info("Annotation over, press 1 to save the result")
            else:
                pass

    @Viewer.bind_key('4')
    def set_label_4(viewer):
        global counter
        if (int(annotation_widget.labels_nb.value) < 4) is False:
            if counter < len(patch[2]) - 1:
                labels_list.append(4)
                counter += 1
                if patch[2][counter].shape[2] > 3:
                    # if the image is 3D, we switch to 3D view and to display the overlay of patch and mask patch
                    viewer.layers.pop()
                    viewer.layers.pop()
                    viewer.add_labels(patch[1][counter].astype("int"))
                    viewer.add_image(patch[2][counter])
                else:
                    # 2D case
                    viewer.layers.pop()
                    viewer.add_image(patch[2][counter])
                print("label 2", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
            elif counter == len(patch[2]) - 1:
                labels_list.append(4)
                if patch[2][counter].shape[2] > 3:
                    viewer.layers.pop()
                    viewer.layers.pop()
                else:
                    viewer.layers.pop()
                counter += 1
                from skimage.io import imread
                viewer.add_image(imread("https://bitbucket.org/koopa31/napari_package_images/raw/"
                                        "a9fda1dd3361880162474cf0b30119b1e188f53c/image_finish.png"))
                print("annotation over", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
                show_info("Annotation over, press 1 to save the result")
            else:
                pass

    @Viewer.bind_key('5')
    def set_label_5(viewer):
        global counter
        if (int(annotation_widget.labels_nb.value) < 5) is False:
            if counter < len(patch[2]) - 1:
                labels_list.append(5)
                counter += 1
                if patch[2][counter].shape[2] > 3:
                    # if the image is 3D, we switch to 3D view and to display the overlay of patch and mask patch
                    viewer.layers.pop()
                    viewer.layers.pop()
                    viewer.add_labels(patch[1][counter].astype("int"))
                    viewer.add_image(patch[2][counter])
                else:
                    # 2D case
                    viewer.layers.pop()
                    viewer.add_image(patch[2][counter])
                print("label 2", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
            elif counter == len(patch[2]) - 1:
                labels_list.append(5)
                if patch[2][counter].shape[2] > 3:
                    viewer.layers.pop()
                    viewer.layers.pop()
                else:
                    viewer.layers.pop()
                counter += 1
                from skimage.io import imread
                viewer.add_image(imread("https://bitbucket.org/koopa31/napari_package_images/raw/"
                                        "a9fda1dd3361880162474cf0b30119b1e188f53c/image_finish.png"))
                print("annotation over", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
                show_info("Annotation over, press 1 to save the result")
            else:
                pass

    @Viewer.bind_key('6')
    def set_label_6(viewer):
        global counter
        if (int(annotation_widget.labels_nb.value) < 6) is False:
            if counter < len(patch[2]) - 1:
                labels_list.append(6)
                counter += 1
                if patch[2][counter].shape[2] > 3:
                    # if the image is 3D, we switch to 3D view and to display the overlay of patch and mask patch
                    viewer.layers.pop()
                    viewer.layers.pop()
                    viewer.add_labels(patch[1][counter].astype("int"))
                    viewer.add_image(patch[2][counter])
                else:
                    # 2D case
                    viewer.layers.pop()
                    viewer.add_image(patch[2][counter])
                print("label 2", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
            elif counter == len(patch[2]) - 1:
                labels_list.append(6)
                if patch[2][counter].shape[2] > 3:
                    viewer.layers.pop()
                    viewer.layers.pop()
                else:
                    viewer.layers.pop()
                counter += 1
                from skimage.io import imread
                viewer.add_image(imread("https://bitbucket.org/koopa31/napari_package_images/raw/"
                                        "a9fda1dd3361880162474cf0b30119b1e188f53c/image_finish.png"))
                print("annotation over", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
                show_info("Annotation over, press 1 to save the result")
            else:
                pass

    @Viewer.bind_key('r')
    def remove_label(viewer):
        global counter
        labels_list.pop()
        viewer.layers.pop()
        counter -= 1
        viewer.add_image(patch[2][counter])
        print("retour en arriere", labels_list)
        viewer.status = str(counter) + " images processed over " + str(len(patch[2]))

    @thread_worker
    def generate_patches(viewer, imagettes_nb, patch_size):
        for im in viewer:
            if "mask" in im.name:
                labels = im.data
                global labels_path
                labels_path = im.source.path
            else:
                image = im.data
                global image_path
                image_path = im.source.path

        half_patch_size = patch_size // 2
        contours_color = [0, 255, 0]

        props = regionprops(labels)
        random.shuffle(props)

        mini_props = props[:imagettes_nb]

        imagettes_list = []
        imagettes_contours_list = []
        maskettes_list = []

        global mini_props_list
        mini_props_list = []
        for i, prop in enumerate(mini_props):
            if prop.area != 0:
                if image.shape[2] <= 3:
                    mini_props_list.append({"centroid": prop.centroid, "coords": prop.coords})
                    imagette = image[int(prop.centroid[0]) - half_patch_size:int(prop.centroid[0]) + half_patch_size,
                               int(prop.centroid[1]) - half_patch_size: int(prop.centroid[1]) + half_patch_size]

                    maskette = np.zeros((imagette.shape[0], imagette.shape[1]))
                    xb = int(prop.centroid[0]) - half_patch_size
                    yb = int(prop.centroid[1]) - half_patch_size
                    if xb >= 0 and yb >= 0 and xb + 2 * half_patch_size + 1 < image.shape[
                        0] and yb + 2 * half_patch_size + 1 < \
                            image.shape[1]:
                        for x, y in prop.coords:
                            maskette[x - xb, y - yb] = 1

                        eroded_mask = cv2.erode(maskette, np.ones((3, 3), np.uint8))
                        contours = maskette - eroded_mask
                        imagette_contours = imagette.copy()
                        imagette_contours[:, :, 0][contours != 0] = contours_color[0]
                        imagette_contours[:, :, 1][contours != 0] = contours_color[1]
                        imagette_contours[:, :, 2][contours != 0] = contours_color[2]

                    imagettes_list.append(imagette)
                    maskettes_list.append(maskette)
                    imagettes_contours_list.append(imagette_contours)

                else:
                    imagette = image[int(prop.centroid[0]) - half_patch_size:int(prop.centroid[0]) + half_patch_size,
                                     int(prop.centroid[1]) - half_patch_size:int(prop.centroid[1]) + half_patch_size,
                                     int(prop.centroid[2]) - half_patch_size:int(prop.centroid[2]) + half_patch_size]

                    maskette = np.zeros((imagette.shape[0], imagette.shape[1], imagette.shape[2]))
                    xb = int(prop.centroid[0]) - half_patch_size
                    yb = int(prop.centroid[1]) - half_patch_size
                    zb = int(prop.centroid[2]) - half_patch_size

                    if xb >= 0 and yb >= 0 and zb >= 0 and xb + 2 * half_patch_size + 1 < image.shape[0] and \
                       yb + 2 * half_patch_size + 1 < image.shape[1] and zb + 2 * half_patch_size + 1 < image.shape[2]:
                        for x, y, z in prop.coords:
                            maskette[x - xb, y - yb, z - zb] = 1
                        imagettes_list.append(imagette)
                        maskettes_list.append(maskette)
                        imagettes_contours_list.append(imagette)

        print(len(imagettes_list))

        global patch
        patch = (imagettes_list, maskettes_list, imagettes_contours_list)
        return patch

    @magicgui(
        layout='vertical',
        patch_size=dict(widget_type='LineEdit', label='patch size', value=200, tooltip='extracted patch size'),
        patch_nb=dict(widget_type='LineEdit', label='patches number', value=10, tooltip='number of extracted patches'),
        labels_nb=dict(widget_type='ComboBox', label='labels number', choices=labels_number, value=2,
                       tooltip='Number of possible labels'),
        extract_pacthes_button=dict(widget_type='PushButton', text='extract patches from image',
                                    tooltip='extraction of patches to be annotated from the segmentation mask'),
    )
    def annotation_widget(  # label_logo,
            viewer: Viewer,
            patch_size,
            patch_nb,
            extract_pacthes_button,
            labels_nb,

    ) -> None:
        # Import when users activate plugin
        return

    def display_first_patch(patch):
        for i in range(0, len(annotation_widget.viewer.value.layers)):
            annotation_widget.viewer.value.layers.pop()
        if patch[2][0].shape[2] > 3:
            # if the image is 3D, we switch to 3D view and to display the overlay of patch and mask patch
            annotation_widget.viewer.value.dims.ndisplay = 3
            annotation_widget.viewer.value.add_labels(patch[1][0].astype("int"))
            annotation_widget.viewer.value.add_image(patch[2][0])
        else:
            # 2D case
            annotation_widget.viewer.value.add_image(patch[2][0])

    @annotation_widget.extract_pacthes_button.changed.connect
    def _extract_patches(e: Any):
        patch_worker = generate_patches(annotation_widget.viewer.value.layers, int(annotation_widget.patch_nb.value),
                                        int(annotation_widget.patch_size.value))
        patch_worker.returned.connect(display_first_patch)
        patch_worker.start()
        print('patch extraction done')

    return annotation_widget


def Training():
    from napari.qt.threading import thread_worker

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def get_image_patch(image, mask, region_props, labels_list, torch_type):
        """
        This function aims at contructing the tensors of the images and their labels
        """

        labels_tensor = torch.from_numpy(labels_list).type(torch_type)
        labels_tensor = nn.functional.one_hot(labels_tensor.type(torch.cuda.LongTensor))

        img_patch_list = []

        for i, position in enumerate(region_props):

            imagette = image[int(region_props[i]["centroid"][0]) - (patch_size//2):int(region_props[i]["centroid"][0])
                             + (patch_size//2), int(region_props[i]["centroid"][1]) - (patch_size//2):
                             int(region_props[i]["centroid"][1]) + (patch_size//2)]

            imagette_mask = np.zeros((imagette.shape[0], imagette.shape[1]))
            xb = int(region_props[i]["centroid"][0]) - (patch_size//2)
            yb = int(region_props[i]["centroid"][1]) - (patch_size//2)

            for x, y in region_props[i]["coords"]:
                imagette_mask[x - xb, y - yb] = 1

            concat_image = np.zeros((imagette.shape[0], imagette.shape[1], 4))
            concat_image[:, :, :3] = imagette
            concat_image[:, :, 3] = imagette_mask
            # Normalization of the image
            concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

            img_patch_list.append(concat_image)

        train_data = CustomDataset(data_list=img_patch_list, labels_tensor=labels_tensor, transform=transform)
        return train_data

    @thread_worker
    def train(image, mask, region_props, labels_list, nn_type, loss_func, lr, epochs_nb, rot, h_flip,
              v_flip, prob, batch_size):

        global transform
        if rot is True and h_flip is False and v_flip is False:
            transform = A.Compose([
                A.Rotate(-90, 90, p=prob),
                ToTensorV2()])
        elif rot is False and h_flip is True and v_flip is False:
            transform = A.Compose([
                A.HorizontalFlip(p=prob),
                ToTensorV2()])
        elif rot is False and h_flip is False and v_flip is True:
            transform = A.Compose([
                A.VerticalFlip(p=prob),
                ToTensorV2()])
        elif rot is True and h_flip is False and v_flip is True:
            transform = A.Compose([
                A.Rotate(-90, 90, p=prob),
                A.VerticalFlip(p=prob),
                ToTensorV2()])
        elif rot is True and h_flip is True and v_flip is False:
            transform = A.Compose([
                A.Rotate(-90, 90, p=prob),
                A.HorizontalFlip(p=prob),
                ToTensorV2()])
        elif rot is True and h_flip is True and v_flip is True:
            transform = A.Compose([
                A.Rotate(-90, 90, p=prob),
                A.HorizontalFlip(p=prob),
                A.VerticalFlip(p=prob),
                ToTensorV2()])
        else:
            transform = A.Compose([ToTensorV2()])

        nn_dict = {"ResNet18": "resnet18", "GoogleNet": "googlenet", "DenseNet": "densenet"}
        # Setting of network
        model = eval("models." + nn_dict[nn_type] + "(pretrained=False)")

        set_parameter_requires_grad(model, True)
        # The fully connected layer of the network is changed so the ouptut size is "labels_number + 1" as we have
        # "labels_number" labels
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, max(labels_list) + 1, bias=True)
        model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        torch_type = torch.cuda.FloatTensor

        losses_dict = {
            "CrossEntropy": "CrossEntropyLoss",
            "L1Smooth": "L1SmoothLoss",
            "BCE": "BceLoss",
            "Distance": "DistanceLoss",
            "L1": "L1_Loss",
            "MSE": "MseLoss"
        }

        # Setting the optimizer
        LR = lr
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True

        # Parameters
        labels_list = np.array(labels_list)

        # Generators
        train_data = get_image_patch(image, mask, region_props, labels_list, torch_type)
        training_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        # Optimizer
        model.to("cuda")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.Adam(params_to_update, lr=LR)

        # Loss function
        LOSS_LIST = []
        weights = np.ones([max(labels_list) + 1])
        weights[0] = 0
        weights = torch.from_numpy(weights)
        loss = eval("nn." + losses_dict[loss_func] + "(weight=weights).type(torch_type)")
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        # Loop over epochs
        iterations_number = epochs_nb
        for epoch in range(iterations_number):
            print("Epoch ", epoch + 1)
            for phase in ["train", "val"]:
                if phase == "train":
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
                        print(total_loss.item())
                        # scheduler.step()
                        if (epoch + 1) % 500 == 0:
                            d = {"model": model, "optimizer_state_dict": optimizer,
                                 "loss": loss, "training_nb": iterations_number, "loss_list": LOSS_LIST}
                            model_path = os.path.join(folder, "training_ABS_full" + str(epoch + 1))
                            if model_path.endswith(".pt") or model_path.endswith(".pth"):
                                torch.save(d, model_path)
                            else:
                                torch.save(d, model_path + ".pth")

                elif phase == "val":
                    pass

    @magicgui(
        auto_call=True,
        layout='vertical',
        load_data_button=dict(widget_type='PushButton', text='Load data', tooltip='Load the image and the labels'),
        lr=dict(widget_type='LineEdit', label='Learning rate', value=0.01, tooltip='Learning rate'),
        nn=dict(widget_type='ComboBox', label='Network architecture', choices=networks_list, value="ResNet18",
                       tooltip='All the available network architectures'),
        loss=dict(widget_type='ComboBox', label='Loss function', choices=losses_list, value="CrossEntropy",
                  tooltip='All the available loss functions'),
        epochs=dict(widget_type='LineEdit', label='Epochs number', value=1000, tooltip='Epochs number'),
        launch_training_button=dict(widget_type='PushButton', text='Launch training', tooltip='Launch training'),
        DATA_AUGMENTATION_TYPE=dict(widget_type='Label'),
        rotations=dict(widget_type='CheckBox', text='Rotations', tooltip='Rotations'),
        v_flip=dict(widget_type='CheckBox', text='Vertical flip', tooltip='Vertical flip'),
        h_flip=dict(widget_type='CheckBox', text='Horizontal flip', tooltip='Horizontal flip'),
        prob=dict(widget_type='LineEdit', label='Probability', value=0.8, tooltip='Probability'),
        b_size=dict(widget_type='LineEdit', label='Batch Size', value=128, tooltip='Batch Size'),
    )
    def training_widget(  # label_logo,
            viewer: Viewer,
            load_data_button,
            nn,
            loss,
            lr,
            epochs,
            b_size,
            DATA_AUGMENTATION_TYPE,
            rotations,
            h_flip,
            v_flip,
            prob,
            launch_training_button,

    ) -> None:
        # Import when users activate plugin
        return

    @training_widget.load_data_button.changed.connect
    def _load_data(e: Any):
        path = QFileDialog.getOpenFileName(None, 'Open File', options=QFileDialog.DontUseNativeDialog)[0]
        """
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        """
        b = torch.load(path)

        global image_path
        global labels_path
        global region_props
        global labels_list
        global image
        global mask
        image_path = b["image_path"]
        labels_path = b["labels_path"]
        region_props = b["regionprops"]
        labels_list = b["labels_list"]

        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(labels_path))
        training_widget.viewer.value.add_image(imread(image_path))
        training_widget.viewer.value.add_labels(imread(labels_path))

        return

    @training_widget.launch_training_button.changed.connect
    def _extract_patches(e: Any):
        training_worker = train(image, mask, region_props, labels_list, training_widget.nn.value,
                                training_widget.loss.value, float(training_widget.lr.value),
                                int(training_widget.epochs.value), training_widget.rotations.value,
                                training_widget.h_flip.value, training_widget.v_flip.value,
                                float(training_widget.prob.value), int(training_widget.b_size.value))
        training_worker.start()
        show_info('Training started')

    return training_widget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [Annotation, Training]
