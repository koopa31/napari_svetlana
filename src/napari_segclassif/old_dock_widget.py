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
from skimage.io import imread, imsave

from .CustomDataset import CustomDataset
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import models
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torchvision import transforms

from joblib import Parallel, delayed

from superqt import ensure_main_thread
from qtpy.QtWidgets import QFileDialog

from line_profiler_pycharm import profile
from .CNN3D import CNN3D
from .CNN2D import CNN2D

from .PredictionDataset import PredictionDataset
from .Prediction3DDataset import Prediction3DDataset
from .PredictionMulti3DDataset import PredictionMulti3DDataset


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
networks_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "AlexNet", "DenseNet121",
                 "DenseNet161", "DenseNet169", "DenseNet201"]
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

            viewer.layers.clear()
            viewer.add_image(patch[0][counter])
            viewer.add_labels(patch[2][counter])
            annotation_widget.viewer.value.layers[1].color = {1: "green"}
            viewer.layers.selection.active = viewer.layers[0]
            if "freeze" in globals() and freeze is True:
                viewer.layers[0].contrast_limits_range = [m, M]

            print("label 1", labels_list)
            viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
        elif counter == len(patch[2]) - 1:
            labels_list.append(1)
            viewer.layers.clear()
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
            torch.save(res_dict, path)

    @Viewer.bind_key('2')
    def set_label_2(viewer):
        global counter
        if counter < len(patch[2]) - 1:
            labels_list.append(2)
            counter += 1

            viewer.layers.clear()
            viewer.add_image(patch[0][counter])
            viewer.add_labels(patch[2][counter])
            annotation_widget.viewer.value.layers[1].color = {1: "green"}
            viewer.layers.selection.active = viewer.layers[0]
            if "freeze" in globals() and freeze is True:
                viewer.layers[0].contrast_limits_range = [m, M]

            print("label 2", labels_list)
            viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
        elif counter == len(patch[2]) - 1:
            labels_list.append(2)
            viewer.layers.clear()

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

                viewer.layers.clear()
                viewer.add_image(patch[0][counter])
                viewer.add_labels(patch[2][counter])
                annotation_widget.viewer.value.layers[1].color = {1: "green"}
                viewer.layers.selection.active = viewer.layers[0]
                if "freeze" in globals() and freeze is True:
                    viewer.layers[0].contrast_limits_range = [m, M]

                print("label 3", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
            elif counter == len(patch[2]) - 1:
                labels_list.append(3)
                viewer.layers.clear()

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

                viewer.layers.clear()
                viewer.add_image(patch[0][counter])
                viewer.add_labels(patch[2][counter])
                annotation_widget.viewer.value.layers[1].color = {1: "green"}
                viewer.layers.selection.active = viewer.layers[0]
                if "freeze" in globals() and freeze is True:
                    viewer.layers[0].contrast_limits_range = [m, M]

                print("label 4", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
            elif counter == len(patch[2]) - 1:
                labels_list.append(4)
                viewer.layers.clear()

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

                viewer.layers.clear()
                viewer.add_image(patch[0][counter])
                viewer.add_labels(patch[2][counter])
                annotation_widget.viewer.value.layers[1].color = {1: "green"}
                viewer.layers.selection.active = viewer.layers[0]
                if "freeze" in globals() and freeze is True:
                    viewer.layers[0].contrast_limits_range = [m, M]

                print("label 5", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
            elif counter == len(patch[2]) - 1:
                labels_list.append(5)
                viewer.layers.clear()

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

                viewer.layers.clear()
                viewer.add_image(patch[0][counter])
                viewer.add_labels(patch[2][counter])
                annotation_widget.viewer.value.layers[1].color = {1: "green"}
                viewer.layers.selection.active = viewer.layers[0]
                if "freeze" in globals() and freeze is True:
                    viewer.layers[0].contrast_limits_range = [m, M]

                print("label 6", labels_list)
                viewer.status = str(counter) + " images processed over " + str(len(patch[2]))
            elif counter == len(patch[2]) - 1:
                labels_list.append(6)
                viewer.layers.clear()

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
        viewer.layers.clear()
        counter -= 1
        viewer.add_image(patch[0][counter])
        viewer.add_labels(patch[2][counter])
        annotation_widget.viewer.value.layers[1].color = {1: "green"}
        viewer.layers.selection.active = viewer.layers[0]
        if "freeze" in globals() and freeze is True:
            viewer.layers[0].contrast_limits_range = [m, M]
        print("retour en arriere", labels_list)
        viewer.status = str(counter) + " images processed over " + str(len(patch[2]))

    @thread_worker
    def generate_patches(viewer, imagettes_nb, patch_size):
        for im in viewer:
            if "mask" in im.name:
                global labels
                labels = im.data
                global labels_path
                labels_path = im.source.path
            else:
                image = im.data
                global image_path
                image_path = im.source.path

        half_patch_size = patch_size // 2
        # contours_color = (0, np.iinfo(image.dtype).max, 0)

        props = regionprops(labels)
        random.shuffle(props)

        global mini_props
        mini_props = props[:imagettes_nb]

        imagettes_list = []
        imagettes_contours_list = []
        maskettes_list = []

        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        # 2D
        if image.shape[2] <= 3:
            pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1), (0, 0)), mode="constant")
            pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
        # Multi-spectral 2D
        elif len(image.shape) == 4:
            pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1), (0, 0),
                                       (patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
            pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

        # Multi-spectral 3D
        elif image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
            pad_image = np.pad(image, ((0, 0), (patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
            pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

        # 3D
        else:
            pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
            pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

        global mini_props_list
        mini_props_list = []
        for i, prop in enumerate(mini_props):
            if prop.area != 0:
                if image.shape[2] <= 3:

                    xmin = (int(prop.centroid[0]) + half_patch_size + 1) - half_patch_size
                    xmax = (int(prop.centroid[0]) + half_patch_size + 1) + half_patch_size
                    ymin = (int(prop.centroid[1]) + half_patch_size + 1) - half_patch_size
                    ymax = (int(prop.centroid[1]) + half_patch_size + 1) + half_patch_size

                    imagette = pad_image[xmin:xmax, ymin:ymax]
                    maskette = pad_labels[xmin:xmax, ymin:ymax].copy()

                    maskette[maskette != prop.label] = 0
                    maskette[maskette == prop.label] = 1

                    eroded_mask = cv2.erode(maskette, np.ones((3, 3), np.uint8))
                    contours = maskette - eroded_mask
                    # imagette_contours = imagette.copy()
                    # imagette_contours[contours != 0] = contours_color

                    imagettes_list.append(imagette)
                    maskettes_list.append(maskette)
                    imagettes_contours_list.append(contours)
                    mini_props_list.append({"centroid": prop.centroid, "coords": prop.coords, "label": prop.label})

                elif image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                    xmin = (int(prop.centroid[0]) + half_patch_size + 1) - half_patch_size
                    xmax = (int(prop.centroid[0]) + half_patch_size + 1) + half_patch_size
                    ymin = (int(prop.centroid[1]) + half_patch_size + 1) - half_patch_size
                    ymax = (int(prop.centroid[1]) + half_patch_size + 1) + half_patch_size

                    imagette = pad_image[:, xmin:xmax, ymin:ymax].copy()
                    maskette = pad_labels[xmin:xmax, ymin:ymax].copy()

                    maskette[maskette != prop.label] = 0
                    maskette[maskette == prop.label] = 1

                    eroded_mask = cv2.erode(maskette, np.ones((3, 3), np.uint8))
                    contours = maskette - eroded_mask
                    # imagette_contours = imagette.copy()
                    # imagette_contours[contours != 0] = contours_color

                    imagettes_list.append(imagette)
                    maskettes_list.append(maskette)
                    imagettes_contours_list.append(contours)
                    mini_props_list.append({"centroid": prop.centroid, "coords": prop.coords, "label": prop.label})

                elif len(image.shape) == 4:

                    xmin = (int(prop.centroid[0]) + half_patch_size + 1) - half_patch_size
                    xmax = (int(prop.centroid[0]) + half_patch_size + 1) + half_patch_size
                    ymin = (int(prop.centroid[1]) + half_patch_size + 1) - half_patch_size
                    ymax = (int(prop.centroid[1]) + half_patch_size + 1) + half_patch_size
                    zmin = (int(prop.centroid[2]) + half_patch_size + 1) - half_patch_size
                    zmax = (int(prop.centroid[2]) + half_patch_size + 1) + half_patch_size

                    imagette = pad_image[xmin:xmax, :, ymin:ymax, zmin:zmax]

                    maskette = pad_labels[xmin:xmax, ymin:ymax, zmin:zmax].copy()

                    maskette[maskette != prop.label] = 0
                    maskette[maskette == prop.label] = 1

                    imagettes_list.append(imagette)
                    maskettes_list.append(maskette)
                    imagettes_contours_list.append(maskette)
                    mini_props_list.append({"centroid": prop.centroid, "coords": prop.coords, "label": prop.label})

                else:
                    # 3D case
                    xmin = (int(prop.centroid[0]) + half_patch_size + 1) - half_patch_size
                    xmax = (int(prop.centroid[0]) + half_patch_size + 1) + half_patch_size
                    ymin = (int(prop.centroid[1]) + half_patch_size + 1) - half_patch_size
                    ymax = (int(prop.centroid[1]) + half_patch_size + 1) + half_patch_size
                    zmin = (int(prop.centroid[2]) + half_patch_size + 1) - half_patch_size
                    zmax = (int(prop.centroid[2]) + half_patch_size + 1) + half_patch_size

                    imagette = pad_image[xmin:xmax, ymin:ymax, zmin:zmax]

                    maskette = pad_labels[xmin:xmax, ymin:ymax, zmin:zmax].copy()

                    maskette[maskette != prop.label] = 0
                    maskette[maskette == prop.label] = 1

                    imagettes_list.append(imagette)
                    maskettes_list.append(maskette)
                    imagettes_contours_list.append(maskette)
                    mini_props_list.append({"centroid": prop.centroid, "coords": prop.coords, "label": prop.label})

        print(len(imagettes_list))

        global patch
        patch = (imagettes_list, maskettes_list, imagettes_contours_list)
        return patch

    @magicgui(
        auto_call=True,
        layout='vertical',
        patch_size=dict(widget_type='LineEdit', label='patch size', value=200, tooltip='extracted patch size'),
        patch_nb=dict(widget_type='LineEdit', label='patches number', value=10, tooltip='number of extracted patches'),
        labels_nb=dict(widget_type='ComboBox', label='labels number', choices=labels_number, value=2,
                       tooltip='Number of possible labels'),
        extract_pacthes_button=dict(widget_type='PushButton', text='extract patches from image',
                                    tooltip='extraction of patches to be annotated from the segmentation mask'),
        estimate_size_button=dict(widget_type='PushButton', text='Estimate patch size',
                                  tooltip='Automatically estimate an optimal patch size'),
        save_regionprops_button=dict(widget_type='PushButton', text='Save objects statistics', tooltip='Save the '
                                                                                                       'properties of the annotated objects in a binary file, loadable using torch.load'),
        generate_im_labs_button=dict(widget_type='PushButton', text='Save masks of labels', tooltip='Save one '
                                                                                                    'per attributed label'),
        fc=dict(widget_type='CheckBox', text='Freeze contrast', tooltip='Freeze contrast'),
    )
    def annotation_widget(  # label_logo,
            viewer: Viewer,
            estimate_size_button,
            patch_size,
            patch_nb,
            extract_pacthes_button,
            labels_nb,
            save_regionprops_button,
            generate_im_labs_button,
            fc,

    ) -> None:
        # Import when users activate plugin
        return

    def display_first_patch(patch):
        for i in range(0, len(annotation_widget.viewer.value.layers)):
            annotation_widget.viewer.value.layers.pop()

        if patch[0][0].shape[2] <= 3 or (patch[0][0].shape[0] < patch[0][0].shape[1]
                                         and patch[0][0].shape[0] < patch[0][0].shape[2]):
            # 2D case
            annotation_widget.viewer.value.add_image(patch[0][0])
            annotation_widget.viewer.value.add_labels(patch[2][0].astype("int"))

            annotation_widget.viewer.value.layers[1].color = {1: "green"}
            annotation_widget.viewer.value.layers.selection.active = annotation_widget.viewer.value.layers[0]
            if "freeze" in globals() and freeze is True:
                annotation_widget.viewer.value.layers[0].contrast_limits_range = [m, M]

        elif len(patch[0][0].shape) == 4:
            annotation_widget.viewer.value.dims.ndisplay = 3
            annotation_widget.viewer.value.add_image(patch[0][0])
            annotation_widget.viewer.value.add_labels(patch[2][0].astype("int"))

            annotation_widget.viewer.value.layers[1].color = {1: "green"}
            annotation_widget.viewer.value.layers.selection.active = annotation_widget.viewer.value.layers[0]
            annotation_widget.viewer.value.layers[0].contrast_limits_range = [m, M]
            if "freeze" in globals() and freeze is True:
                annotation_widget.viewer.value.layers[0].contrast_limits_range = [m, M]

        else:
            # if the image is 3D, we switch to 3D view and to display the overlay of patch and mask patch
            annotation_widget.viewer.value.dims.ndisplay = 3
            annotation_widget.viewer.value.add_image(patch[0][0])
            annotation_widget.viewer.value.add_labels(patch[2][0].astype("int"))

            annotation_widget.viewer.value.layers[1].color = {1: "green"}
            annotation_widget.viewer.value.layers.selection.active = annotation_widget.viewer.value.layers[0]
            if "freeze" in globals() and freeze is True:
                annotation_widget.viewer.value.layers[0].contrast_limits_range = [m, M]

    @annotation_widget.fc.changed.connect
    def freeze_contrast(e: Any):
        global freeze
        if e is True:
            global m, M
            freeze = True
            for im in annotation_widget.viewer.value.layers:
                if "mask" not in im.name and im._type_string == "image":
                    m = im.contrast_limits[0]
                    M = im.contrast_limits[1]
        else:
            freeze = False

    @annotation_widget.extract_pacthes_button.changed.connect
    def _extract_patches(e: Any):
        patch_worker = generate_patches(annotation_widget.viewer.value.layers, int(annotation_widget.patch_nb.value),
                                        int(annotation_widget.patch_size.value))
        patch_worker.returned.connect(display_first_patch)
        patch_worker.start()
        print('patch extraction done')

    @annotation_widget.estimate_size_button.changed.connect
    def estimate_patch_size():
        for im in annotation_widget.viewer.value.layers:
            if "mask" in im.name:
                labels = im.data

        props = regionprops(labels)
        x = sorted(props, key=lambda r: r.area, reverse=True)

        # Cas 2D/3D
        if len(labels.shape) == 2:
            xmax = x[0].bbox[2] - x[0].bbox[0]
            ymax = x[0].bbox[3] - x[0].bbox[1]
            length = max(xmax, ymax)
        else:
            xmax = x[0].bbox[3] - x[0].bbox[0]
            ymax = x[0].bbox[4] - x[0].bbox[1]
            zmax = x[0].bbox[5] - x[0].bbox[2]
            length = max(xmax, ymax, zmax)

        patch_size = int(length + 0.4 * length)

        annotation_widget.patch_size.value = patch_size

        # Affichage nombre max de patch qu'on peut extraire
        annotation_widget.patch_nb.label = "patches number (" + str(len(props)) + " max)"
        annotation_widget.patch_nb.value = len(props)

    @annotation_widget.save_regionprops_button.changed.connect
    def save_regionprops():
        if counter != len(mini_props):
            raise ValueError("Please finish your annotation before saving the stats")
        else:
            path = QFileDialog.getSaveFileName(None, 'Save File', options=QFileDialog.DontUseNativeDialog)[0]
            props_list = []

            if mini_props[0].coords.shape[1] == 3:
                for i, prop in enumerate(mini_props):
                    props_list.append({"position": prop.label, "coords": prop.coords, "centroid": prop.centroid,
                                       "area": prop.area, "label": int(labels_list[i])})
            else:
                for i, prop in enumerate(mini_props):
                    props_list.append({"position": prop.label, "coords": prop.coords, "centroid": prop.centroid,
                                       "eccentricity": prop.eccentricity, "area": prop.area,
                                       "perimeter": prop.perimeter,
                                       "label": int(labels_list[i])})
            torch.save(props_list, path)

    @annotation_widget.generate_im_labs_button.changed.connect
    def generate_im_labels():
        im_labs_list = []
        # We create as many images as labels
        for i in range(0, max(labels_list)):
            im_labs_list.append(np.zeros_like(labels).astype(np.uint16))

        if len(labels.shape) == 3:
            for i, prop in enumerate(mini_props):
                im_labs_list[labels_list[i] - 1][prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = prop.label
        else:
            for i, prop in enumerate(mini_props):
                im_labs_list[labels_list[i] - 1][prop.coords[:, 0], prop.coords[:, 1]] = prop.label

        for i, im in enumerate(im_labs_list):
            imsave(os.path.splitext(labels_path)[0] + "_label" + str(i + 1) + ".tif", im)

    return annotation_widget


def Training():
    from napari.qt.threading import thread_worker

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def get_image_patch(image, labels, region_props, labels_list, torch_type, case):
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

        for i, position in enumerate(region_props):
            if case == "2D" or case == "multi_2D":
                xmin = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) - (patch_size // 2)
                xmax = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) + (patch_size // 2)
                ymin = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) - (patch_size // 2)
                ymax = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) + (patch_size // 2)

                imagette = image[xmin:xmax, ymin:ymax].copy()
                imagette_mask = labels[xmin:xmax, ymin:ymax].copy()

                imagette_mask[imagette_mask != region_props[i]["label"]] = 0
                imagette_mask[imagette_mask == region_props[i]["label"]] = max_type_val

                concat_image = np.zeros((imagette.shape[0], imagette.shape[1], image.shape[2] + 1))
                concat_image[:, :, :-1] = imagette
                concat_image[:, :, -1] = imagette_mask
                # Normalization of the image
                concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

                img_patch_list.append(concat_image)
            elif case == "multi_3D":
                xmin = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) - (patch_size // 2)
                xmax = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) + (patch_size // 2)
                ymin = (int(region_props[i]["centroid"][2]) + (patch_size // 2) + 1) - (patch_size // 2)
                ymax = (int(region_props[i]["centroid"][2]) + (patch_size // 2) + 1) + (patch_size // 2)
                zmin = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) - (patch_size // 2)
                zmax = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) + (patch_size // 2)

                imagette = image[:, xmin:xmax, ymin:ymax, zmin:zmax].copy()

                imagette_mask = labels[xmin:xmax, ymin:ymax, zmin:zmax].copy()

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

    def train(viewer, image, mask, region_props, labels_list, nn_type, loss_func, lr, epochs_nb, rot, h_flip,
              v_flip, prob, batch_size, saving_ep, training_name, model=None):

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

        nn_dict = {"ResNet18": "resnet18", "ResNet34": "resnet34", "ResNet50": "resnet50", "ResNet101": "resnet101",
                   "ResNet152": "resnet152", "AlexNet": "alexnet", "DenseNet121": "densenet121",
                   "DenseNet161": "densenet161", "DenseNet169": "densenet169", "DenseNet201": "densenet201"}
        # Setting of network

        if model is None:
            # 2D case
            if image.shape[2] <= 3:
                case = "2D"
                model = eval("models." + nn_dict[nn_type] + "(pretrained=False)")
                set_parameter_requires_grad(model, True)

                if "resnet" in nn_dict[nn_type]:
                    # The fully connected layer of the network is changed so the ouptut size is "labels_number + 1" as we have
                    # "labels_number" labels
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
                    model.features[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

            elif image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                case = "multi_2D"
                model = eval("models." + nn_dict[nn_type] + "(pretrained=False)")
                set_parameter_requires_grad(model, True)
                image = np.transpose(image, (1, 2, 0))

                if "resnet" in nn_dict[nn_type]:
                    # The fully connected layer of the network is changed so the ouptut size is "labels_number + 1" as we have
                    # "labels_number" labels
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, max(labels_list) + 1, bias=True)
                    model.conv1 = nn.Conv2d(image.shape[2] + 1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
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

            elif len(image.shape) == 4:
                case = "multi_3D"
                image = np.transpose(image, (1, 2, 3, 0))
                mask = np.transpose(mask, (1, 2, 0))
                model = CNN3D(max(labels_list), image.shape[0] + 1)

            # 3D case
            else:
                case = "3D"
                model = CNN3D(max(labels_list), 2)

        else:
            if image.shape[2] <= 3:
                case = "2D"
                model = CNN2D(max(labels_list), 4)

            elif image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                case = "multi_2D"

            elif len(image.shape) == 4:
                case = "multi_3D"

            # 3D case
            else:
                case = "3D"

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
        train_data = get_image_patch(pad_image, pad_labels, region_props, labels_list, torch_type, case)
        training_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

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
        loss = eval("nn." + losses_dict[loss_func] + "(weight=weights).type(torch_type)")
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        # Loop over epochs
        iterations_number = epochs_nb
        # folder where to save the training
        save_folder = os.path.split(image_path)[0]
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
                                total_loss = loss(out, local_labels.type(torch.cuda.FloatTensor))
                                optimizer.zero_grad()
                                total_loss.backward()
                                optimizer.step()
                                LOSS_LIST.append(total_loss.item())
                                print(total_loss.item())
                                viewer.value.status = "loss = " + str(total_loss.item())
                                # scheduler.step()
                                if (epoch + 1) % saving_ep == 0:
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

                                found = True

                        elif phase == "val":
                            pass
                    yield epoch + 1

            except:
                if "bs" in locals():
                    bs -= 1
                    bs = 2 ** (np.floor(np.log(bs) / np.log(2)))
                else:
                    bs = 2 ** (np.floor(np.log(batch_size) / np.log(2)))
                training_loader = DataLoader(dataset=train_data, batch_size=int(bs), shuffle=True)

        """
        folder = "/home/cazorla/Images/TEST"
        model = torch.load(os.path.join(folder, "training_ABS_full1000.pth"))["model"]
        from natsort import natsorted
        validation_folder = "/home/cazorla/Images/TEST/Imagettes_validation"
        validation_imagettes_list = natsorted(
            [f for f in os.listdir(validation_folder) if os.path.isfile(os.path.join(validation_folder, f)) and
             os.path.join(validation_folder, f).endswith(".png") and os.path.splitext(f)[0].endswith("imagette")])
        validation_imagettes_masks_list = natsorted(
            [f for f in os.listdir(validation_folder) if os.path.isfile(os.path.join(validation_folder, f)) and
             os.path.join(validation_folder, f).endswith(".png") and os.path.splitext(f)[0].endswith("mask")])
        model.eval()
        values_list = []
        LIST = []
        for image_nb in range(0, len(validation_imagettes_list)):
            imagette = np.array(Image.open(os.path.join(validation_folder, validation_imagettes_list[image_nb])))
            imagette_mask = np.array(Image.open(os.path.join(validation_folder,
                                     validation_imagettes_masks_list[image_nb])))

            concat_image = np.zeros((imagette.shape[0], imagette.shape[1], 4))
            concat_image[:, :, :3] = imagette
            concat_image[:, :, 3] = imagette_mask
            # Normalization of the image
            concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

            img_t = transforms.Compose([transforms.ToTensor()])(concat_image)
            batch_t = torch.unsqueeze(img_t, 0).type(torch.float32).to("cuda")
            out = model(batch_t)
            _, index = torch.max(out, 1)
            print(index)
        """
        plt.plot(LOSS_LIST)
        plt.show()

    @magicgui(
        auto_call=True,
        layout='vertical',
        load_data_button=dict(widget_type='PushButton', text='Load data', tooltip='Load the image and the labels'),
        lr=dict(widget_type='LineEdit', label='Learning rate', value=0.01, tooltip='Learning rate'),
        nn=dict(widget_type='ComboBox', label='Network architecture', choices=networks_list, value="ResNet18",
                tooltip='All the available network architectures'),
        load_custom_model_button=dict(widget_type='PushButton', text='Load custom NN',
                                      tooltip='Load your own NN pretrained or not'),
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
        SAVING_PARAMETERS=dict(widget_type='Label'),
        saving_ep=dict(widget_type='LineEdit', label='Save training each (epochs)', value=100,
                       tooltip='Each how many epoch the training should be saved'),
        training_name=dict(widget_type='LineEdit', label='Training file name', tooltip='Training file name'),

    )
    def training_widget(  # label_logo,
            viewer: Viewer,
            load_data_button,
            nn,
            load_custom_model_button,
            loss,
            lr,
            epochs,
            b_size,
            DATA_AUGMENTATION_TYPE,
            rotations,
            h_flip,
            v_flip,
            prob,
            SAVING_PARAMETERS,
            saving_ep,
            training_name,
            launch_training_button,

    ) -> None:
        # Import when users activate plugin
        return

    @training_widget.load_data_button.changed.connect
    def _load_data(e: Any):
        training_widget.viewer.value.layers.clear()
        path = QFileDialog.getOpenFileName(None, 'Open File', options=QFileDialog.DontUseNativeDialog)[0]

        b = torch.load(path)

        global image_path
        global labels_path
        global region_props
        global labels_list
        global patch_size
        global image
        global mask
        image_path = b["image_path"]
        labels_path = b["labels_path"]
        region_props = b["regionprops"]
        labels_list = b["labels_list"]
        patch_size = int(b["patch_size"])

        image = imread(image_path)
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        mask = imread(labels_path)
        training_widget.viewer.value.add_image(image)
        training_widget.viewer.value.add_labels(mask)

        return

    @training_widget.load_custom_model_button.changed.connect
    def load_custom_model():
        path = QFileDialog.getOpenFileName(None, 'Open File', options=QFileDialog.DontUseNativeDialog)[0]
        checkpoint = torch.load(path)
        global model
        model = checkpoint["model"]
        show_info("Model loaded successfully")

    @training_widget.launch_training_button.changed.connect
    def _launch_training(e: Any):
        if "model" in globals():
            training_worker = thread_worker(train, progress={"total": int(training_widget.epochs.value)}) \
                (training_widget.viewer, image, mask, region_props, labels_list, training_widget.nn.value,
                 training_widget.loss.value, float(training_widget.lr.value),
                 int(training_widget.epochs.value), training_widget.rotations.value,
                 training_widget.h_flip.value, training_widget.v_flip.value,
                 float(training_widget.prob.value), int(training_widget.b_size.value),
                 int(training_widget.saving_ep.value), str(training_widget.training_name.value), model)
        else:

            training_worker = thread_worker(train, progress={"total": int(training_widget.epochs.value)})(
                training_widget.viewer, image, mask, region_props, labels_list,
                training_widget.nn.value,
                training_widget.loss.value, float(training_widget.lr.value),
                int(training_widget.epochs.value), training_widget.rotations.value,
                training_widget.h_flip.value, training_widget.v_flip.value,
                float(training_widget.prob.value), int(training_widget.b_size.value),
                int(training_widget.saving_ep.value), str(training_widget.training_name.value), None)

        training_worker.start()
        show_info('Training started')

    return training_widget


def Prediction():
    from napari.qt.threading import thread_worker

    def draw_predicted_contour(compteur, prop, imagette_contours, i, list_pred):

        imagette_contours[prop.coords[:, 0], prop.coords[:, 1]] = list_pred[i].item()
        if list_pred[i] == 1:
            compteur += 1
        return compteur

    def draw_3d_prediction(compteur, prop, imagette_contours, i, list_pred):

        imagette_contours[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = list_pred[i].item()
        if list_pred[i] == 1:
            compteur += 1
        return compteur

    def predict(image, labels, props, patch_size, batch_size):

        import time
        start = time.time()
        compteur = 0
        global imagette_contours

        try:
            max = np.iinfo(image.dtype).max
        except:
            max = np.finfo(image.dtype).max

        if image.shape[2] <= 3 or (image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]):
            if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                image = np.transpose(image, (1, 2, 0))

            imagette_contours = np.zeros((image.shape[0], image.shape[1]))
            pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1), (0, 0)), mode="constant")
            pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

            data = PredictionDataset(pad_image, pad_labels, props, patch_size // 2, max)

        elif len(image.shape) == 4:
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
            yield i + 1

        show_info("Prediction of patches done, please wait while the result image is being generated...")
        if len(labels.shape) == 2:
            compteur = Parallel(n_jobs=-1, require="sharedmem")(
                delayed(draw_predicted_contour)(compteur, prop, imagette_contours, i, list_pred)
                for i, prop in enumerate(props))
        else:
            compteur = Parallel(n_jobs=-1, require="sharedmem")(
                delayed(draw_3d_prediction)(compteur, prop, imagette_contours, i, list_pred)
                for i, prop in enumerate(props))

        # Deletion of the old mask
        prediction_widget.viewer.value.layers.pop()

        stop = time.time()
        print("temps de traitement", stop - start)
        show_info(str(np.sum(compteur)) + " objects remaining over " + str(len(props)))
        print(str(np.sum(compteur)) + " objects remaining over " + str(len(props)))

        return imagette_contours.astype(np.uint8)

    @magicgui(
        auto_call=True,
        layout='vertical',
        batch_size=dict(widget_type='LineEdit', label='Batch size', value=100, tooltip='Batch size'),
        load_data_button=dict(widget_type='PushButton', text='Load data', tooltip='Load the image and the labels'),
        launch_prediction_button=dict(widget_type='PushButton', text='Launch prediction', tooltip='Launch prediction'),
        bound=dict(widget_type='CheckBox', text='Show boundaries only', tooltip='Show boundaries only'),
        generate_im_labs_button=dict(widget_type='PushButton', text='Save masks of labels', tooltip='Save one '
                                                                                                    'per attributed label'),
        save_regionprops_button=dict(widget_type='PushButton', text='Save objects statistics', tooltip='Save the '
                                                                                                       'properties of the annotated objects in a binary file, loadable using torch.load'),
    )
    def prediction_widget(  # label_logo,
            viewer: Viewer,
            load_data_button,
            batch_size,
            launch_prediction_button,
            bound,
            save_regionprops_button,
            generate_im_labs_button,

    ) -> None:
        # Import when users activate plugin
        return

    @prediction_widget.load_data_button.changed.connect
    def _load_data(e: Any):
        # Removal of the remaining images of the previous widgets
        prediction_widget.viewer.value.layers.clear()
        path = QFileDialog.getOpenFileName(None, 'Open File', options=QFileDialog.DontUseNativeDialog)[0]
        """
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        """
        b = torch.load(path)

        global image
        global mask
        global model
        global patch_size
        global labels_path

        image_path = b["image_path"]
        labels_path = b["labels_path"]
        model = b["model"].to("cuda")
        patch_size = b["patch_size"]
        model.eval()

        image = imread(image_path)
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        mask = imread(labels_path)
        prediction_widget.viewer.value.add_image(image)
        prediction_widget.viewer.value.add_labels(mask)

        return

    def display_result(image):
        prediction_widget.viewer.value.add_labels(image)
        # Chose of the colours
        prediction_widget.viewer.value.layers[1].name = "Classified labels"
        if len(np.unique(prediction_widget.viewer.value.layers[1].data)) == 3:
            prediction_widget.viewer.value.layers[1].color = {1: "green", 2: "red"}

    @prediction_widget.launch_prediction_button.changed.connect
    def _launch_prediction(e: Any):
        global props
        props = regionprops(mask)
        prediction_worker = thread_worker(predict, progress={
            "total": int(np.ceil(len(props) / int(prediction_widget.batch_size.value)))}) \
            (image, mask, props, patch_size, int(prediction_widget.batch_size.value))
        # Addition of the new labels
        prediction_worker.returned.connect(display_result)

        prediction_worker.start()
        show_info('Prediction started')

    @prediction_widget.bound.changed.connect
    def show_boundaries(e: Any):

        if "edge_im" not in globals():
            # computation of the cells segmentation edges
            eroded_contours = cv2.erode(np.uint16(mask), np.ones((7, 7), np.uint8))
            eroded_labels = mask - eroded_contours

            # Removing the inside of the cells in the binary result using the edges mask computed just before
            global edge_im
            edge_im = imagette_contours.copy().astype(np.uint8)
            edge_im[eroded_labels == 0] = 0
        if e is True:
            prediction_widget.viewer.value.layers.pop()
            prediction_widget.viewer.value.add_labels(edge_im)
            if len(np.unique(prediction_widget.viewer.value.layers[1].data)) == 3:
                prediction_widget.viewer.value.layers[1].color = {1: "green", 2: "red"}
        else:
            prediction_widget.viewer.value.layers.pop()
            prediction_widget.viewer.value.add_labels(imagette_contours.astype(np.uint8))
            if len(np.unique(prediction_widget.viewer.value.layers[1].data)) == 3:
                prediction_widget.viewer.value.layers[1].color = {1: "green", 2: "red"}

    @prediction_widget.save_regionprops_button.changed.connect
    def save_regionprops():

        path = QFileDialog.getSaveFileName(None, 'Save File', options=QFileDialog.DontUseNativeDialog)[0]
        props_list = []
        if len(mask.shape) == 3:
            for i, prop in enumerate(props):
                props_list.append({"position": prop.label, "coords": prop.coords, "centroid": prop.centroid,
                                   "area": prop.area, "label": int(list_pred[i].item())})
        else:
            for i, prop in enumerate(props):
                props_list.append({"position": prop.label, "coords": prop.coords, "centroid": prop.centroid,
                                   "eccentricity": prop.eccentricity, "area": prop.area, "perimeter": prop.perimeter,
                                   "label": int(list_pred[i].item())})
        torch.save(props_list, path)

    @prediction_widget.generate_im_labs_button.changed.connect
    def generate_im_labels():
        im_labs_list = []
        # We create as many images as labels
        for i in range(0, max(list_pred)):
            im_labs_list.append(np.zeros_like(mask).astype(np.uint16))

        if len(mask.shape) == 3:
            for i, prop in enumerate(props):
                im_labs_list[list_pred[i] - 1][prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = prop.label
        else:
            for i, prop in enumerate(props):
                im_labs_list[list_pred[i] - 1][prop.coords[:, 0], prop.coords[:, 1]] = prop.label

        for i, im in enumerate(im_labs_list):
            imsave(os.path.splitext(labels_path)[0] + "_label" + str(i + 1) + ".tif", im)

    return prediction_widget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [Annotation, Training, Prediction]
