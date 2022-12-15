"""
Svetlana dock widget module
"""
import platform
import subprocess

# We call a bash function to install torch for windows as we cannot install the Cuda version from the setup.cfg file

p1 = subprocess.Popen("ltt install torch torchvision torchaudio grad-cam==1.4.6", shell=True)

try:
    import torch
except ImportError:
    print("Please wait while PyTorch dependencies are being installed")
    p1.wait()

import functools
import os
import pickle
import random
from typing import Any

import cv2
import dask.dataframe
import matplotlib.pyplot as plt
from napari_plugin_engine import napari_hook_implementation
from napari.utils.notifications import show_info, notification_manager

import time
import numpy as np
import json

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

# from line_profiler_pycharm import profile
from .XP.modulable_CNN3D import CNN3D
from .CNN2D import CNN2D

from .PredictionDataset import PredictionDataset, max_to_1, min_max_norm
from .Prediction3DDataset import Prediction3DDataset
from .PredictionMulti3DDataset import PredictionMulti3DDataset

# import gradcam
from .Grad_Cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image
from pytorch_grad_cam import GuidedBackpropReLUModel

from PIL import ImageDraw, ImageFont, Image
import pandas as pd

if torch.cuda.is_available() is True:
    try:
        import cupy as cu
        from cucim.skimage.morphology import dilation, ball, disk
        cuda = True
    except ImportError:
        from skimage.morphology import ball, dilation, disk
        cuda = False
        # Not necessary for other OS as Cucim is only compatible with Linux
        if platform.system() == 'Linux':
            show_info("Could not make cupy work, please do: conda install cudatoolkit=10.2")
else:
    from skimage.morphology import ball, dilation, disk
    cuda = False


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


labels_number = [('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9)]
networks_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "AlexNet", "DenseNet121",
                 "DenseNet161", "DenseNet169", "DenseNet201", "lightNN_2_3", "lightNN_3_5", "lightNN_4_5"]
dims_list = ["2D", "3D"]
losses_list = ["CrossEntropy", "L1Smooth", "BCE", "Distance", "L1", "MSE"]
data_norm_list = ["min max normalization", "max to 1 normalization", "no normalization"]

counter = 0
# counter of images to be annotated
image_counter = 0
total_counter = 0
# edges thickness
thickness = 7

global_labels_list = []
global_im_path_list = []
global_lab_path_list = []
global_mini_props_list = []

# This variable forbids to label if a correct object has not been clicked before
enable_labeling = True

# list of all regionprops to be exported in a specific binary file
props_to_be_saved = []

retrain = False
loaded_network = None

double_click = False
heatmap = False


def Annotation():
    """
    Annotation plugin code
    @return:
    """
    from napari.qt.threading import thread_worker

    def on_pressed(key):
        """
        When a key between 1 and 9 is pressed, it calls the function which sets the label
        @param key: int between 1 and 9
        @return:
        """

        def set_label(viewer):
            """
            Attributes a label to an object you want to annotate
            @param viewer: Napari viewer instance
            @return:
            """
            global counter, total_counter, enable_labeling, double_click
            if enable_labeling is True:
                print("key is ", key)
                if (int(annotation_widget.labels_nb.value) < key) is False:
                    if counter < len(props) - 1:
                        global_labels_list[image_counter].append(key - 1)

                        global_mini_props_list[image_counter].append(
                            {"centroid": props[indexes[counter]].centroid, "coords": props[indexes[counter]].coords,
                             "label": props[indexes[counter]].label})
                        if case == "2D" or case == "multi2D":
                            progression_mask[
                                props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1]] = key
                            # If show labels is ticked, then we refresh it on the fly as we label
                            if annotation_widget.show_labs.value is True:
                                annotation_widget.viewer.value.layers.pop()
                                show_labs(True)

                            counter += 1
                            total_counter += 1

                            # focus on the next object to annotate
                            if double_click is False:
                                #viewer.camera.zoom = zoom_factor
                                zoom_factor = viewer.camera.zoom
                                viewer.camera.center = (0, int(props[indexes[counter]].centroid[0]),
                                                        int(props[indexes[counter]].centroid[1]))
                                viewer.camera.zoom = zoom_factor + 10 ** -8
                            # deletion of the old contours and drawing of the new one
                            circle_mask[circle_mask != 0] = 0
                            circle_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1]] = 1
                            eroded_contours = cv2.erode(np.uint16(circle_mask), np.ones((5, 5), np.uint8))
                            eroded_labels = circle_mask - eroded_contours

                            # Pyramidal representation of the contours to enhance the display speed

                            annotation_widget.viewer.value.layers["object to annotate"].data_raw[0][
                                annotation_widget.viewer.value.layers["object to annotate"].data_raw[0] != 0] = 0
                            annotation_widget.viewer.value.layers["object to annotate"].data_raw[0][eroded_labels == 1] = 1

                            for i in range(1, len(annotation_widget.viewer.value.layers["object to annotate"].data_raw)):
                                annotation_widget.viewer.value.layers["object to annotate"].data_raw[i] = \
                                    cv2.resize(annotation_widget.viewer.value.layers["object to annotate"].data_raw[0], (
                                        annotation_widget.viewer.value.layers["object to annotate"].data_raw[0].shape[
                                            0] // 2 ** i,
                                        annotation_widget.viewer.value.layers["object to annotate"].data_raw[0].shape[
                                            1] // 2 ** i))

                        else:
                            progression_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1],
                                             props[indexes[counter]].coords[:, 2]] = key
                            # If show labels is ticked, then we refresh it on the fly as we label
                            if annotation_widget.show_labs.value is True:
                                annotation_widget.viewer.value.layers.pop()
                                show_labs(True)

                            counter += 1
                            total_counter += 1

                            # focus on the next object to annotate
                            if double_click is False:
                                #viewer.camera.zoom = zoom_factor
                                zoom_factor = viewer.camera.zoom
                                viewer.camera.center = (int(props[indexes[counter]].centroid[0]),
                                                        int(props[indexes[counter]].centroid[1]),
                                                        int(props[indexes[counter]].centroid[2]))
                                annotation_widget.viewer.value.dims.current_step = (
                                    int(props[indexes[counter]].centroid[0]),
                                    annotation_widget.viewer.value.dims.current_step[
                                        1],
                                    annotation_widget.viewer.value.dims.current_step[
                                        2])
                                viewer.camera.zoom = zoom_factor + 10 ** -8
                            # deletion of the old contours and drawing of the new one
                            circle_mask[circle_mask != 0] = 0
                            circle_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1],
                                        props[indexes[counter]].coords[:, 2]] = 1
                            annotation_widget.viewer.value.layers["object to annotate"].data = circle_mask

                        annotation_widget.viewer.value.layers.selection.active = annotation_widget.viewer.value.layers[
                            image_layer_name]

                        print("label 1", global_labels_list[image_counter])
                        viewer.status = str(counter) + " images processed over " + str(len(props)) + " (" + \
                                        str(total_counter) + " over the whole batch)"
                    elif counter == len(props) - 1:
                        global_labels_list[image_counter].append(key - 1)
                        global_mini_props_list[image_counter].append(
                            {"centroid": props[indexes[counter]].centroid, "coords": props[indexes[counter]].coords,
                             "label": props[indexes[counter]].label})
                        progression_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1]] = \
                            key
                        if annotation_widget.show_labs.value is True:
                            annotation_widget.viewer.value.layers.pop()
                            show_labs(True)
                        counter += 1
                        total_counter += 1

                        print("annotation over", global_labels_list[image_counter])
                        viewer.status = str(counter) + " images processed over " + str(len(props)) + " (" + \
                                        str(total_counter) + " over the whole batch)"
                        show_info("Image entirely annotated")
                    else:
                        show_info("Image entirely annotated")
                # If click to annotate is activated, we don't want the user to be able to click two labels in a row
                # without re-choosing a new object
                if double_click is True:
                    enable_labeling = False
        return set_label

    @Viewer.bind_key('r', overwrite=True)
    def remove_label(viewer):
        """
        Cancels the last attributed label when r is pressed and goes back to previous object to be re-labelled
        @param viewer: Napari viewer instance
        @return:
        """
        global counter, total_counter
        global_labels_list[image_counter].pop()
        global_mini_props_list[image_counter].pop()
        counter -= 1
        total_counter -= 1

        if case == "2D" or case == "multi2D":
            progression_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1]] = 0

            # If show labels is ticked, then we refresh it on the fly as we label
            if annotation_widget.show_labs.value is True:
                annotation_widget.viewer.value.layers.pop()
                show_labs(True)

            if double_click is False:
                #viewer.camera.zoom = zoom_factor
                zoom_factor = viewer.camera.zoom
                viewer.camera.center = (0, int(props[indexes[counter]].centroid[0]),
                                        int(props[indexes[counter]].centroid[1]))
                viewer.camera.zoom = zoom_factor + 10 ** -8
            circle_mask[circle_mask != 0] = 0
            circle_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1]] = 1
            eroded_contours = cv2.erode(np.uint16(circle_mask), np.ones((5, 5), np.uint8))
            eroded_labels = circle_mask - eroded_contours

            # Pyramidal representation of the contours to enhance the display speed
            annotation_widget.viewer.value.layers["object to annotate"].data_raw[0][
                annotation_widget.viewer.value.layers["object to annotate"].data_raw[0] != 0] = 0
            annotation_widget.viewer.value.layers["object to annotate"].data_raw[0][eroded_labels == 1] = 1

            for i in range(1, len(annotation_widget.viewer.value.layers["object to annotate"].data_raw)):
                annotation_widget.viewer.value.layers["object to annotate"].data_raw[i] = \
                    cv2.resize(annotation_widget.viewer.value.layers["object to annotate"].data_raw[0], (
                        annotation_widget.viewer.value.layers["object to annotate"].data_raw[0].shape[0] // 2 ** i,
                        annotation_widget.viewer.value.layers["object to annotate"].data_raw[0].shape[1] // 2 ** i))

        else:
            progression_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1],
                             props[indexes[counter]].coords[:, 2]] = 0

            # If show labels is ticked, then we refresh it on the fly as we label
            if annotation_widget.show_labs.value is True:
                annotation_widget.viewer.value.layers.pop()
                show_labs(True)

            if double_click is False:
                #viewer.camera.zoom = zoom_factor
                zoom_factor = viewer.camera.zoom
                viewer.camera.center = (int(props[indexes[counter]].centroid[0]),
                                        int(props[indexes[counter]].centroid[1]),
                                        int(props[indexes[counter]].centroid[2]))
                annotation_widget.viewer.value.dims.current_step = (int(props[indexes[counter]].centroid[0]),
                                                                    annotation_widget.viewer.value.dims.current_step[
                                                                        1],
                                                                    annotation_widget.viewer.value.dims.current_step[
                                                                        2])
                viewer.camera.zoom = zoom_factor + 10 ** -8
            circle_mask[circle_mask != 0] = 0
            circle_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1],
                        props[indexes[counter]].coords[:, 2]] = 1
            annotation_widget.viewer.value.layers["object to annotate"].data = circle_mask

        annotation_widget.viewer.value.layers.selection.active = annotation_widget.viewer.value.layers[
            image_layer_name]

        print("retour en arriere", global_labels_list[image_counter])
        viewer.status = str(counter) + " images processed over " + str(len(props)) + " (" + \
                        str(total_counter) + " over the whole batch)"

    @thread_worker
    def generate_patches(viewer, patch_size):
        """
        Computes the correct zoom factor to focus on objects to annotate
        @param viewer: Napari vewer instance
        @param patch_size: the estimated optimal patch size
        @return: the zoom factor and the regionprops computed from the labels mask
        """
        """for im in viewer:
            if "mask" in im.name:
                global labels
                labels = im.data
                global labels_path
                labels_path = im.source.path
            elif im.name != "confidence map" and "previous prediction":
                image = im.data
                global image_path
                image_path = im.source.path"""
        global labels, labels_path, image_path
        labels = mask.copy()
        image = Image.copy()
        labels_path = mask_path_list[int(annotation_widget.image_index_button.value) - 1]
        image_path = image_path_list[int(annotation_widget.image_index_button.value) - 1]

        global case, zoom_factor

        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)

        if image.shape[2] <= 3:
            case = "2D"
            zoom_factor = old_zoom * (max(image.shape[0], image.shape[1]) / patch_size)
        elif len(image.shape) == 4:
            case = "multi3D"
            zoom_factor = image.shape[2] / patch_size
        else:
            zoom_factor = image.shape[1] / patch_size
            case = yield
            while case is None:
                pass
            print(case)

        global props, props_to_be_saved
        props = regionprops(labels)

        # The regionprops are shuffled to be randomly labeled
        global indexes
        indexes = np.random.permutation(np.arange(0, len(props))).tolist()

        present = []
        for prop in props_to_be_saved:
            # Check if they are identical, converting the two props into pandas frames and checking if True is the only
            # value inside the comparison table
            if len(props) == len(prop["props"]):
                if len(np.unique(pd.DataFrame(prop["props"]) == pd.DataFrame(props))) == 1:
                    present.append(True)
                else:
                    present.append(False)
            else:
                present.append(False)

        if True not in present:
            props_to_be_saved.append({"props": props, "indexes": indexes})

        return props, zoom_factor

    def send_case_to_thread():
        global case
        patch_worker.pause()
        from .CustomDialog import CustomDialog
        diag = CustomDialog()
        diag.exec()
        case = diag.get_case()
        patch_worker.send(diag.get_case())
        patch_worker.resume()

    @magicgui(
        auto_call=False,
        call_button=False,
        layout='vertical',
        load_images_button=dict(widget_type='PushButton', text='LOAD IMAGES',
                                tooltip='Load images'),
        OR=dict(widget_type='Label'),
        restart_labelling_button=dict(widget_type='PushButton', text='RESUME LABELLING',
                                      tooltip='Resume a started labelling'),
        vertical_space1=dict(widget_type='Label', label=' '),
        image_choice=dict(widget_type='Label', label="IMAGE CHOICE"),
        vertical_space2=dict(widget_type='Label', label=' '),
        patch_size_choice=dict(widget_type='Label', label="PATCH SIZE CHOICE"),
        vertical_space3=dict(widget_type='Label', label=' '),
        vertical_space4=dict(widget_type='Label', label=' '),
        next_button=dict(widget_type='PushButton', text='Next image',
                         tooltip='Next image'),
        previous_button=dict(widget_type='PushButton', text='Previous image',
                             tooltip='Previous image'),
        image_index_button=dict(widget_type='LineEdit', label='Image index', value=1,
                                tooltip='Image index in the batch'),
        patch_size=dict(widget_type='LineEdit', label='patch size', value=200, tooltip='extracted patch size'),
        labels_nb=dict(widget_type='ComboBox', label='labels number', choices=labels_number, value=2,
                       tooltip='Number of possible labels'),
        extract_pacthes_button=dict(widget_type='PushButton', text='Start annotation',
                                    tooltip='Start annotation'),
        estimate_size_button=dict(widget_type='PushButton', text='Estimate patch size',
                                  tooltip='Automatically estimate an optimal patch size'),
        save_button=dict(widget_type='PushButton', text='Save annotation', tooltip='Save annotation', enabled=False),

        save_regionprops_button=dict(widget_type='PushButton', text='Save objects statistics', tooltip='Save the '
                                     'properties of the annotated objects in a binary file, loadable using torch.load',
                                     enabled=False),
        generate_im_labs_button=dict(widget_type='PushButton', text='Save masks of labels', tooltip='Save one '
                                     'per attributed label', enabled=False),
        show_labs=dict(widget_type='CheckBox', text='Show labeled objects', tooltip='Show labeled objects', enabled=False),
        click_annotate=dict(widget_type='CheckBox', text='Click to annotate', tooltip='Click to annotate', enabled=False),
        youtube_button=dict(widget_type='PushButton', text="", tooltip="Youtube tutorial"),
        doc_button=dict(widget_type='PushButton', text="", tooltip="Documentation"),
    )
    def annotation_widget(  # label_logo,
            viewer: Viewer,
            load_images_button,
            OR,
            restart_labelling_button,
            vertical_space1,
            image_choice,
            previous_button,
            image_index_button,
            next_button,
            vertical_space2,
            patch_size_choice,
            estimate_size_button,
            patch_size,
            vertical_space3,
            extract_pacthes_button,
            labels_nb,
            vertical_space4,
            save_button,
            save_regionprops_button,
            generate_im_labs_button,
            show_labs,
            click_annotate,
            youtube_button,
            doc_button

    ) -> None:
        # Create a black image just so layer variable exists
        # This global instance of the viewer is created to be able to display images from the prediction plugin when
        # it's being opened
        global V
        V = viewer

        if len(viewer.layers) == 0:
            viewer.add_image(np.zeros((1000, 1000)))
        # Import when users activate plugin
        global layer, double_click
        # By default, we do not annotate clicking
        double_click = False

        layer = viewer.layers["Image"]

        # We generate the functions to add a label when a key i pressed
        for i in range(1, 10):
            viewer.bind_key(str(i), on_pressed(i), overwrite=True)

        @layer.mouse_double_click_callbacks.append
        def label_clicking(layer, event):
            """
            When click to annotate option is activated, retrieves the coordinate of the clicked object to give him a
            label
            @param layer:
            @param event: Qt click event
            @return:
            """
            global case, enable_labeling
            if double_click is True:
                if case == "2D":
                    ind = labels[int(event.position[0]), int(event.position[1])] - 1
                elif case == "multi2D":
                    ind = labels[int(event.position[1]), int(event.position[2])] - 1
                elif case == "3D" or case == "multi3D":
                    ind = labels[int(event.position[0]), int(event.position[1]), int(event.position[2])] - 1

                try:
                    indexes.remove(ind)
                    indexes.insert(counter, ind)
                    print('position', event.position)
                    show_info("Choose a label for that object")
                    enable_labeling = True
                except ValueError:
                    show_info("please click on a valid object")
                    enable_labeling = False

    from qtpy.QtGui import QIcon
    icon = QIcon("../webinar.png")
    annotation_widget.youtube_button.native.setIcon(icon)
    annotation_widget.youtube_button.native.setStyleSheet("QPushButton { border: none; }")
    annotation_widget.youtube_button.native.setText("YOUTUBE TUTORIAL")
    icon = QIcon("../doc.png")
    annotation_widget.doc_button.native.setIcon(icon)
    annotation_widget.doc_button.native.setStyleSheet("QPushButton { border: none; }")
    annotation_widget.doc_button.native.setText("DOCUMENTATION")

    annotation_widget.show()

    @annotation_widget.youtube_button.changed.connect
    def launch_tutorial(e: Any):
        import webbrowser
        webbrowser.open("www.youtube.com")

    @annotation_widget.doc_button.changed.connect
    def launch_doc(e: Any):
        import webbrowser
        webbrowser.open("https://svetlana-documentation.readthedocs.io/en/latest/")

    @annotation_widget.load_images_button.changed.connect
    def load_images(e: Any):
        """
        Loads the folder containing the images to annotates and displays the first one
        @param e:
        @return:
        """

        # Make sure buttons are greyed so we cannot click them before launching annotation
        annotation_widget.save_button.enabled = False
        annotation_widget.save_regionprops_button.enabled = False
        annotation_widget.generate_im_labs_button.enabled = False
        annotation_widget.show_labs.enabled = False
        annotation_widget.click_annotate.enabled = False

        # Gets the folder url and the two subfolder containing the images and the masks
        global images_folder, masks_folder, parent_path, counter, total_counter, case, props_to_be_saved
        props_to_be_saved = []
        case = None

        # As autocall is set to False, it is necessary to call the function when loading the data
        annotation_widget.viewer.value.layers.clear()

        # Make sure they are reset to True in case another batch has been processed before, so you can reset the batch
        # size too
        annotation_widget.estimate_size_button.enabled = True
        annotation_widget.patch_size.enabled = True

        # Choice of the batch folder
        parent_path = QFileDialog.getExistingDirectory(None, 'Choose the parent folder which contains folders Images '
                                                             'and Masks',
                                                       options=QFileDialog.DontUseNativeDialog)

        images_folder = os.path.join(parent_path, "Images")
        masks_folder = os.path.join(parent_path, "Masks")

        # Check if the selected folder is correct
        if os.path.isdir(images_folder) is True and os.path.isdir(masks_folder) is True:

            # Gets the list of images and masks
            global image_path_list, mask_path_list, global_im_path_list, global_lab_path_list, global_labels_list, \
                global_mini_props_list, mini_props_list, Image, mask

            image_path_list = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder)])
            mask_path_list = sorted([os.path.join(masks_folder, f) for f in os.listdir(masks_folder)])
            global_im_path_list = image_path_list.copy()
            global_lab_path_list = mask_path_list.copy()

            # reset of these lists when loading new dataset
            global_labels_list = []
            global_mini_props_list = []

            # labels counters reset
            counter = 0
            total_counter = 0

            for i in range(0, len(image_path_list)):
                global_labels_list.append([])
                global_mini_props_list.append([])

            # Deletion of remaining image and displaying of the first uimage of the list
            annotation_widget.viewer.value.layers.clear()
            # If image is 3D multichannel, it is splitted into several 3D images
            Image = imread(image_path_list[image_counter])
            mask = imread(mask_path_list[image_counter])
            if len(Image.shape) == 4:
                annotation_widget.viewer.value.add_image(Image, channel_axis=1, name="Image")
            else:
                annotation_widget.viewer.value.add_image(imread(image_path_list[image_counter]))
            annotation_widget.viewer.value.add_labels(mask)
            annotation_widget.viewer.value.layers[-1].name = "mask"

            # original zoom factor to correct when annotating
            global old_zoom
            old_zoom = annotation_widget.viewer.value.camera.zoom

            # Must be called at the end of loading data so the layer for labeling bay double clicking can be defined as
            # the layer named Image
            annotation_widget()
        else:
            show_info("ERROR: The folder should contain two folders called Images and Masks")

    @annotation_widget.restart_labelling_button.changed.connect
    def restart_labelling(e: Any):
        """
        Function to resume the labelling after a first prediction or not
        """

        # Make sure buttons are greyed so we cannot click them before launching annotation
        annotation_widget.save_button.enabled = False
        annotation_widget.save_regionprops_button.enabled = False
        annotation_widget.generate_im_labs_button.enabled = False
        annotation_widget.show_labs.enabled = False
        annotation_widget.click_annotate.enabled = False

        global images_folder, masks_folder, parent_path, image_path_list, mask_path_list, global_im_path_list, \
            global_lab_path_list, global_labels_list, global_mini_props_list, mini_props_list, counter, \
            image_counter, patch_size, pred_path_list, total_counter, conf_path_list, case, props_to_be_saved, Image, \
            mask

        props_to_be_saved = []

        case = None

        # As autocall is set to False, it is necessary to call the function when loading the data
        annotation_widget.viewer.value.layers.clear()

        parent_path = QFileDialog.getExistingDirectory(None, 'Choose the parent folder which contains folders Images '
                                                             'and Masks', options=QFileDialog.DontUseNativeDialog)

        images_folder = os.path.join(parent_path, "Images")
        masks_folder = os.path.join(parent_path, "Masks")

        # Check if the selected folder is correct
        if os.path.isdir(images_folder) is True and os.path.isdir(masks_folder) is True:

            labels_file = torch.load(os.path.join(parent_path, "Svetlana", "labels"))
            global_im_path_list = labels_file["image_path"]
            image_path_list = global_im_path_list.copy()
            global_lab_path_list = labels_file["labels_path"]
            mask_path_list = global_lab_path_list.copy()
            if os.path.isdir(os.path.join(parent_path, "Predictions")) is True:
                pred_path_list = sorted([os.path.join(parent_path, "Predictions", f) for f in
                                         os.listdir(os.path.join(parent_path, "Predictions"))])

            if os.path.isdir(os.path.join(parent_path, "Confidence")) is True:
                conf_path_list = sorted([os.path.join(parent_path, "Confidence", f) for f in
                                         os.listdir(os.path.join(parent_path, "Confidence"))])

            global_labels_list = labels_file["labels_list"]
            global_mini_props_list = labels_file["regionprops"]
            patch_size = labels_file["patch_size"]

            annotation_widget.patch_size.value = patch_size
            annotation_widget.image_index_button.value = 1

            # Disabling the estimate size and patch size in the gui so it is not change while annotating
            annotation_widget.estimate_size_button.enabled = False
            annotation_widget.patch_size.enabled = False

            image_counter = int(annotation_widget.image_index_button.value) - 1

            counter = len(global_labels_list[image_counter])
            total_counter = 0
            for l in global_labels_list:
                total_counter += len(l)

            # Deletion of remaining image and displaying of the first image of the list
            annotation_widget.viewer.value.layers.clear()
            # If image is 3D multichannel, it is splitted into several 3D images
            Image = imread(global_im_path_list[image_counter])
            mask = imread(global_lab_path_list[image_counter])
            if len(Image.shape) == 4:
                annotation_widget.viewer.value.add_image(Image, channel_axis=1, name="Image")
            else:
                annotation_widget.viewer.value.add_image(imread(image_path_list[image_counter]))
            annotation_widget.viewer.value.add_labels(mask)
            annotation_widget.viewer.value.layers[-1].name = "mask"

            if os.path.isdir(os.path.join(parent_path, "Predictions")) is True:
                annotation_widget.viewer.value.add_labels(imread(pred_path_list[image_counter]))
                annotation_widget.viewer.value.layers[2].name = "previous prediction"
                if len(np.unique(annotation_widget.viewer.value.layers["previous prediction"].data)) == 3:
                    annotation_widget.viewer.value.layers["previous prediction"].color = {1: "green", 2: "red"}

            if os.path.isdir(os.path.join(parent_path, "Confidence")) is True and \
                    len(os.listdir(os.path.join(parent_path, "Confidence"))) != 0:
                annotation_widget.viewer.value.add_image(imread(conf_path_list[image_counter]).astype("uint8"))
                annotation_widget.viewer.value.layers[-1].name = "confidence map"

            # original zoom factor to correct when annotating
            global old_zoom
            old_zoom = annotation_widget.viewer.value.camera.zoom

            # Must be called at the end of loading data so the layer for labeling bay double clicking can be defined as
            # the layer named Image
            annotation_widget()

        else:
            show_info("ERROR: The folder should contain two folders called Images and Masks")

    @annotation_widget.image_index_button.changed.connect
    def set_image_index(e: Any):
        """
        Function to set the index of the image to visualize in the batch instead of using previous/next buttons
        """

        global counter, image_counter, mask

        if int(e) > len(global_im_path_list):
            show_info("Too high index")
            annotation_widget.image_index_button.value = len(global_im_path_list)
        elif int(e) < 1:
            show_info("Too low index")
            annotation_widget.image_index_button.value = 1
        else:

            # Make sure buttons are greyed so we cannot click them before launching annotation
            annotation_widget.save_button.enabled = False
            annotation_widget.save_regionprops_button.enabled = False
            annotation_widget.generate_im_labs_button.enabled = False
            annotation_widget.show_labs.enabled = False
            annotation_widget.click_annotate.enabled = False

            image_counter = int(annotation_widget.image_index_button.value) - 1

            counter = len(global_labels_list[image_counter])

            annotation_widget.viewer.value.layers.clear()

            # If image is 3D multichannel, it is splitted into several 3D images
            Image = imread(os.path.join(images_folder, image_path_list[image_counter]))
            mask = imread(os.path.join(masks_folder, mask_path_list[image_counter]))
            if len(Image.shape) == 4:
                annotation_widget.viewer.value.add_image(Image, channel_axis=1, name="Image")
            else:
                annotation_widget.viewer.value.add_image(imread(image_path_list[image_counter]))
            annotation_widget.viewer.value.add_labels(mask)
            annotation_widget.viewer.value.layers[-1].name = "mask"

            if "pred_path_list" in globals():
                annotation_widget.viewer.value.add_labels(imread(pred_path_list[image_counter]))
                annotation_widget.viewer.value.layers[2].name = "previous prediction"
                if len(np.unique(annotation_widget.viewer.value.layers["previous prediction"].data)) == 3:
                    annotation_widget.viewer.value.layers["previous prediction"].color = {1: "green", 2: "red"}

            if "conf_path_list" in globals() and len(os.listdir(os.path.join(parent_path, "Confidence"))) != 0:
                annotation_widget.viewer.value.add_image(imread(conf_path_list[image_counter]).astype("uint8"))
                annotation_widget.viewer.value.layers[-1].name = "confidence map"

            # Must be called at the end of loading data so the layer for labeling bay double clicking can be defined as
            # the layer named Image
            annotation_widget()

    @annotation_widget.next_button.changed.connect
    def next_image(e: Any):
        """
        Loads the next image to annotate
        @param e:
        @return:
        """
        global image_counter, counter, labels_list, mini_props_list, mask

        if image_counter < len(global_im_path_list) - 1:

            # Make sure buttons are greyed so we cannot click them before launching annotation
            annotation_widget.save_button.enabled = False
            annotation_widget.save_regionprops_button.enabled = False
            annotation_widget.generate_im_labs_button.enabled = False
            annotation_widget.show_labs.enabled = False
            annotation_widget.click_annotate.enabled = False

            # Reinitialization of counter for next image
            counter = len(global_labels_list[image_counter + 1])

            image_counter += 1
            # Update of the image index
            annotation_widget.image_index_button.value = image_counter + 1

            annotation_widget.viewer.value.layers.clear()

            # If image is 3D multichannel, it is splitted into several 3D images
            Image = imread(os.path.join(images_folder, image_path_list[image_counter]))
            mask = imread(os.path.join(masks_folder, mask_path_list[image_counter]))
            if len(Image.shape) == 4:
                annotation_widget.viewer.value.add_image(Image, channel_axis=1, name="Image")
            else:
                annotation_widget.viewer.value.add_image(imread(image_path_list[image_counter]))
            annotation_widget.viewer.value.add_labels(mask)
            annotation_widget.viewer.value.layers[-1].name = "mask"

            if "pred_path_list" in globals():
                annotation_widget.viewer.value.add_labels(imread(pred_path_list[image_counter]))
                annotation_widget.viewer.value.layers[2].name = "previous prediction"
                if len(np.unique(annotation_widget.viewer.value.layers["previous prediction"].data)) == 3:
                    annotation_widget.viewer.value.layers["previous prediction"].color = {1: "green", 2: "red"}

            if "conf_path_list" in globals() and len(os.listdir(os.path.join(parent_path, "Confidence"))) != 0:
                annotation_widget.viewer.value.add_image(imread(conf_path_list[image_counter]).astype("uint8"))
                annotation_widget.viewer.value.layers[-1].name = "confidence map"

            # Must be called at the end of loading data so the layer for labeling bay double clicking can be defined as
            # the layer named Image
            annotation_widget()
        else:
            show_info("No more images")

    @annotation_widget.previous_button.changed.connect
    def previous_image(e: Any):
        """
        Loads the next image to annotate
        @param e:
        @return:
        """
        global image_counter, counter, labels_list, mini_props_list, mask
        if image_counter > 0:

            # Make sure buttons are greyed so we cannot click them before launching annotation
            annotation_widget.save_button.enabled = False
            annotation_widget.save_regionprops_button.enabled = False
            annotation_widget.generate_im_labs_button.enabled = False
            annotation_widget.show_labs.enabled = False
            annotation_widget.click_annotate.enabled = False

            image_counter -= 1
            # Update of the image index
            annotation_widget.image_index_button.value = image_counter + 1

            annotation_widget.viewer.value.layers.clear()
            # If image is 3D multichannel, it is splitted into several 3D images
            Image = imread(os.path.join(images_folder, image_path_list[image_counter]))
            mask = imread(os.path.join(masks_folder, mask_path_list[image_counter]))
            if len(Image.shape) == 4:
                annotation_widget.viewer.value.add_image(Image, channel_axis=1, name="Image")
            else:
                annotation_widget.viewer.value.add_image(imread(image_path_list[image_counter]))
            annotation_widget.viewer.value.add_labels(mask)
            annotation_widget.viewer.value.layers[-1].name = "mask"

            if "pred_path_list" in globals():
                annotation_widget.viewer.value.add_labels(imread(pred_path_list[image_counter]))
                annotation_widget.viewer.value.layers[2].name = "previous prediction"
                if len(np.unique(annotation_widget.viewer.value.layers["previous prediction"].data)) == 3:
                    annotation_widget.viewer.value.layers["previous prediction"].color = {1: "green", 2: "red"}

            if "conf_path_list" in globals() and len(os.listdir(os.path.join(parent_path, "Confidence"))) != 0:
                annotation_widget.viewer.value.add_image(imread(conf_path_list[image_counter]).astype("uint8"))
                annotation_widget.viewer.value.layers[-1].name = "confidence map"

            # Reinitialization of counter for next image
            counter = len(global_labels_list[image_counter])

            # Must be called at the end of loading data so the layer for labeling bay double clicking can be defined as
            # the layer named Image
            annotation_widget()

        else:
            show_info("No previous image")

    @annotation_widget.click_annotate.changed.connect
    def click_to_annotate(e: Any):
        """
        Activates click to annotate option
        @param e: boolean value of the checkbox
        @return:
        """
        global double_click, enable_labeling
        if e is True:
            double_click = True
            # select the image so the user can click on it
            annotation_widget.viewer.value.layers.selection.active = annotation_widget.viewer.value.layers[
                image_layer_name]
            enable_labeling = False
        else:
            double_click = False
            enable_labeling = True
        return double_click

    def display_first_patch(x):
        """
        Once generate patches is done, it focuses on the first object of the list to be labelled
        @param x: list containing the output of generate pacthes function (props, zoom factor)
        @return:
        """
        # the first object to annotate is focused
        current_index = indexes[counter]
        props = x[0]
        zoom_factor = x[1]

        if case == "2D" or case == "multi2D":
            annotation_widget.viewer.value.camera.center = (0, props[current_index].centroid[0],
                                                            props[current_index].centroid[1])
        else:
            # annotation_widget.viewer.value.dims.ndisplay = 3
            annotation_widget.viewer.value.camera.center = (props[current_index].centroid[0],
                                                            props[current_index].centroid[1],
                                                            props[current_index].centroid[2])
            annotation_widget.viewer.value.dims.current_step = (int(props[current_index].centroid[0]),
                                                                annotation_widget.viewer.value.dims.current_step[1],
                                                                annotation_widget.viewer.value.dims.current_step[2])

        annotation_widget.viewer.value.camera.zoom = zoom_factor

        global circle_mask, circle_layer_name, image_layer_name, progression_mask
        for layer in annotation_widget.viewer.value.layers:
            if "mask" in layer.name:
                circle_layer_name = layer.name
                circle_mask = np.zeros_like(layer.data)
                # Creation of progression mask if there are already annotation in previous image
                if counter != 0:
                    progression_mask = np.zeros_like(annotation_widget.viewer.value.layers["mask"].data)
                    if case == '2D':
                        for ind, prop in enumerate(global_mini_props_list[image_counter]):
                            progression_mask[prop["coords"][:, 0], prop["coords"][:, 1]] = \
                                global_labels_list[image_counter][ind] + 1
                    elif case == "multi2D":
                        for ind, prop in enumerate(global_mini_props_list[image_counter]):
                            progression_mask[prop["coords"][:, 0], prop["coords"][:, 1]] = \
                                global_labels_list[image_counter][ind] + 1
                    elif case == "3D" or case == "multi3D":
                        for ind, prop in enumerate(global_mini_props_list[image_counter]):
                            progression_mask[prop["coords"][:, 0], prop["coords"][:, 1], prop["coords"][:, 2]] = \
                                global_labels_list[image_counter][ind] + 1
                else:
                    progression_mask = np.zeros_like(circle_mask)
            else:
                image_layer_name = "Image"

        # Contour of object to annotate
        if case == "2D" or case == "multi2D":
            circle_mask[props[current_index].coords[:, 0], props[current_index].coords[:, 1]] = 1
            eroded_contours = cv2.erode(np.uint16(circle_mask), np.ones((5, 5), np.uint8))
            eroded_labels = (circle_mask - eroded_contours).astype(np.uint8)
            # Pyramidal representation of the contours to enhance the display speed
            pyramid = [eroded_labels]
            for i in range(1, 6):
                pyramid.append(cv2.resize(eroded_labels, (eroded_labels.shape[0] // 2 ** i,
                                                          eroded_labels.shape[1] // 2 ** i)))
            annotation_widget.viewer.value.add_labels(pyramid)
            annotation_widget.viewer.value.layers[-1].name = "object to annotate"
        else:
            circle_mask[props[current_index].coords[:, 0], props[current_index].coords[:, 1],
                        props[current_index].coords[:, 2]] = 1
            annotation_widget.viewer.value.add_labels(circle_mask)
            annotation_widget.viewer.value.layers[-1].name = "object to annotate"
            """
            pyramid = [circle_mask]
            for i in range(1, 6):
                pyramid.append(cv2.resize(circle_mask, (circle_mask.shape[0] // 2**i,
                                                        circle_mask.shape[1] // 2**i,
                                                        circle_mask.shape[2] // 2**i)))

            annotation_widget.viewer.value.add_labels(pyramid)"""

        annotation_widget.viewer.value.layers[-1].color = {1: "green"}
        annotation_widget.viewer.value.layers.selection.active = annotation_widget.viewer.value.layers[image_layer_name]

    @annotation_widget.show_labs.changed.connect
    def show_labs(e: Any):
        """
        Shows the mask of the labelled objects
        @param e: boolean value of the checkbox
        @return:
        """
        # CREATION OF PYRAMIDAL MASK TO ACCELERATE DISPLAY
        if case == "2D" or case == "multi2D":
            pyramid = [progression_mask]
            for i in range(1, 6):
                pyramid.append(cv2.resize(progression_mask.astype("uint8"), (progression_mask.shape[0] // 2 ** i,
                                                             progression_mask.shape[1] // 2 ** i)))

            if e is True:
                annotation_widget.viewer.value.add_labels(pyramid, name="progression_mask")
            else:
                annotation_widget.viewer.value.layers.remove("progression_mask")
        else:
            if e is True:
                annotation_widget.viewer.value.add_labels(progression_mask, name="progression_mask")
            else:
                annotation_widget.viewer.value.layers.remove("progression_mask")
        annotation_widget.viewer.value.layers.selection.active = annotation_widget.viewer.value.layers["Image"]

    @annotation_widget.save_button.changed.connect
    def save_annotations(e: Any):
        """
        Saves the information needed for the training in a binary file using Pytorch
        @param e: indicates if the button has been clicked
        @return:
        """

        res_dict = {"image_path": global_im_path_list, "labels_path": global_lab_path_list,
                    "regionprops": global_mini_props_list, "labels_list": global_labels_list,
                    "patch_size": annotation_widget.patch_size.value}
        if os.path.isdir(os.path.join(parent_path, "Svetlana")) is False:
            os.mkdir(os.path.join(parent_path, "Svetlana"))
        torch.save(res_dict, os.path.join(parent_path, "Svetlana", "labels"))
        show_info("Labels saved in folder called Svetlana")

    @annotation_widget.extract_pacthes_button.changed.connect
    def _extract_patches(e: Any):
        """
        Function triggered by the button to start the annotation
        @param e:
        @return:
        """

        # As button are disabled while labelling isn't started, we must make sure the buttons are enabled
        annotation_widget.save_button.enabled = True
        annotation_widget.save_regionprops_button.enabled = True
        annotation_widget.generate_im_labs_button.enabled = True
        annotation_widget.show_labs.enabled = True
        annotation_widget.click_annotate.enabled = True

        global patch_worker
        patch_worker = generate_patches(annotation_widget.viewer.value.layers, int(annotation_widget.patch_size.value))
        patch_worker.returned.connect(display_first_patch)
        patch_worker.yielded.connect(send_case_to_thread)
        patch_worker.start()
        print('patch extraction done')

        # Disabling the estimate size and patch size in the gui so it is not change while annotating
        annotation_widget.estimate_size_button.enabled = False
        annotation_widget.patch_size.enabled = False
        # We make the labels mask invisible so it does not bother not annotate
        annotation_widget.viewer.value.layers["mask"].visible = False

    @annotation_widget.estimate_size_button.changed.connect
    def estimate_patch_size():
        """
        Function triggered by the button to estimate the optimal squared visualization window size around the object
        to annotate
        @return:
        """

        # Regionprops over all the masks so the size si computed using the biggest object of the whole dataset
        props = []
        for p in mask_path_list:
            labels = imread(p)
            props += regionprops(labels)

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

    @annotation_widget.save_regionprops_button.changed.connect
    def save_regionprops():
        """
        Saves the properties of the labelled connected components in a binary file using Pytorch
        @return:
        """

        path = os.path.join(parent_path, "Svetlana", "regionprops")
        props_list = []

        for j, p in enumerate(props_to_be_saved):
            props = p["props"]
            indexes = p["indexes"]
            counter = len(global_labels_list[j])
            props_list.append([])
            # Image name added to the list
            props_list[j].append(global_im_path_list[j])

            if props[:counter + 1][0].coords.shape[1] == 3:
                for i in range(0, counter):
                    props_list[j].append({"position": props[indexes[i]].label, "coords": props[indexes[i]].coords,
                                          "centroid": props[indexes[i]].centroid, "area": props[indexes[i]].area,
                                          "label": int(global_labels_list[j][i])})
            else:
                for i in range(0, counter):
                    props_list[j].append({"position": props[indexes[i]].label, "coords": props[indexes[i]].coords,
                                          "centroid": props[indexes[i]].centroid,
                                          "eccentricity": props[indexes[i]].eccentricity,
                                          "area": props[indexes[i]].area,
                                          "perimeter": props[indexes[i]].perimeter,
                                          "label": int(global_labels_list[j][i])})
        torch.save(props_list, path)
        show_info("ROI properties saved")

    @annotation_widget.generate_im_labs_button.changed.connect
    def generate_im_labels():
        """
        Saves a mask for each class
        @return:
        """
        im_labs_list = []
        # We create as many images as labels
        for i in range(0, max(global_labels_list[image_counter])):
            im_labs_list.append(np.zeros_like(labels).astype(np.uint16))

        props = props_to_be_saved[image_counter]["props"]
        indexes = props_to_be_saved[image_counter]["indexes"]
        counter = len(global_labels_list[image_counter])

        if len(labels.shape) == 3:
            for i in range(0, counter):
                im_labs_list[global_labels_list[image_counter][i] - 1][props[indexes[i]].coords[:, 0],
                                                                       props[indexes[i]].coords[:, 1],
                                                                       props[indexes[i]].coords[:, 2]] = props[
                    indexes[i]].label
        else:
            for i in range(0, counter):
                im_labs_list[global_labels_list[image_counter][i] - 1][props[indexes[i]].coords[:, 0],
                                                                       props[indexes[i]].coords[:, 1]] = \
                    props[indexes[i]].label

        for i, im in enumerate(im_labs_list):
            imsave(os.path.splitext(global_lab_path_list[image_counter])[0] + "_label" + str(i + 1) + ".tif", im)

    return annotation_widget


def Training():
    """
    Training plugin code
    @return:
    """
    from napari.qt.threading import thread_worker

    # We check if parent_path exists, i.e. whe check if a labeling has been performed in the same instance of Svetlana.
    # If it does exists, then there is no need to use the load data button and the labels and directly pre-loaded.

    if "parent_path" in globals():
        path = os.path.join(parent_path, "Svetlana", "labels")
        b = torch.load(path)
        global image_path_list, labels_path_list, region_props_list, labels_list, patch_size, image, mask, \
            config_dict, case

        if "image_path" in b.keys() and "labels_path" in b.keys() and "regionprops" in b.keys() and "labels_list" \
                in b.keys() and "patch_size" in b.keys():

            image_path_list = b["image_path"]
            labels_path_list = b["labels_path"]
            region_props_list = b["regionprops"]
            labels_list = b["labels_list"]
            patch_size = int(b["patch_size"])

            image = imread(image_path_list[0])
            if len(image.shape) == 2:
                image = np.stack((image,) * 3, axis=-1)
            mask = imread(labels_path_list[0])
            #training_widget.viewer.value.add_image(image)
            #training_widget.viewer.value.add_labels(mask)
            # Load parameters from config file
            try:
                init = os.path.join(os.path.split(os.path.split(np.__file__)[0])[0], "napari_svetlana")
                with open(os.path.join(init, 'Config.json'), 'r') as f:
                    config_dict = json.load(f)
            except FileNotFoundError:
                with open(os.path.join(os.getcwd(), 'Config.json'), 'r') as f:
                    config_dict = json.load(f)

            # Copy of config file to folder Svetlana
            save_folder = os.path.join(os.path.split(os.path.split(image_path_list[0])[0])[0], "Svetlana")
            if os.path.isdir(save_folder) is False:
                os.mkdir(save_folder)
            import shutil
            if os.path.exists(os.path.join(save_folder, "Config.json")) is False:

                try:
                    shutil.copy(os.path.join(init, 'Config.json'),
                                os.path.join(save_folder, "Config.json"))
                except FileNotFoundError:
                    shutil.copy(os.path.join(os.getcwd(), 'Config.json'),
                                os.path.join(save_folder, "Config.json"))


    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def get_image_patch(image_list, mask_list, region_props_list, labels_list, torch_type, case, norm_type):
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

        try:
            import cupy as cu
        except ImportError:
            show_info("WARNING: If you want to accelerate computation using Cupy, do in this Conda env: conda install"
                      " cudatoolkit=10.2")

        labels_tensor = torch.from_numpy(labels_list).type(torch_type)
        labels_tensor = nn.functional.one_hot(labels_tensor.type(torch.cuda.LongTensor))

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
                if case == "2D" or case == "multi2D":
                    xmin = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) - (patch_size // 2)
                    xmax = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) + (patch_size // 2)
                    ymin = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) - (patch_size // 2)
                    ymax = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) + (patch_size // 2)

                    imagette = image[xmin:xmax, ymin:ymax].copy()
                    imagette_mask = labels[xmin:xmax, ymin:ymax].copy()

                    imagette_mask[imagette_mask != region_props[i]["label"]] = 0
                    imagette_mask[imagette_mask == region_props[i]["label"]] = 1

                    # dilation of mask
                    if json.loads(config_dict["options"]["dilation"]["dilate_mask"].lower()) is True:

                        if cuda is True:
                            str_el = cu.asarray(disk(int(config_dict["options"]["dilation"]["str_element_size"])))
                            imagette_mask = cu.asnumpy(dilation(cu.asarray(imagette_mask), str_el))
                        else:
                            str_el = disk(int(config_dict["options"]["dilation"]["str_element_size"]))
                            imagette_mask = dilation(imagette_mask, str_el)

                        if case == "2D":
                            imagette *= imagette_mask[:, :, None]
                        else:
                            imagette *= imagette_mask[None, :, :]

                    concat_image = np.zeros((imagette.shape[0], imagette.shape[1], imagette.shape[2] + 1))

                    if norm_type == "min max normalization":
                        imagette = min_max_norm(imagette)
                    elif norm_type == "max to 1 normalization":
                        imagette = max_to_1(imagette)

                    concat_image[:, :, :-1] = imagette
                    concat_image[:, :, -1] = imagette_mask

                    img_patch_list.append(concat_image)
                elif case == "multi3D":
                    xmin = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) - (patch_size // 2)
                    xmax = (int(region_props[i]["centroid"][1]) + (patch_size // 2) + 1) + (patch_size // 2)
                    ymin = (int(region_props[i]["centroid"][2]) + (patch_size // 2) + 1) - (patch_size // 2)
                    ymax = (int(region_props[i]["centroid"][2]) + (patch_size // 2) + 1) + (patch_size // 2)
                    zmin = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) - (patch_size // 2)
                    zmax = (int(region_props[i]["centroid"][0]) + (patch_size // 2) + 1) + (patch_size // 2)

                    imagette = image[:, xmin:xmax, ymin:ymax, zmin:zmax].copy()

                    imagette_mask = labels[xmin:xmax, ymin:ymax, zmin:zmax].copy()

                    imagette_mask[imagette_mask != region_props[i]["label"]] = 0
                    imagette_mask[imagette_mask == region_props[i]["label"]] = 1

                    # dilation of mask
                    if json.loads(config_dict["options"]["dilation"]["dilate_mask"].lower()) is True:

                        if cuda is True:
                            str_el = cu.asarray(ball(int(config_dict["options"]["dilation"]["str_element_size"])))
                            imagette_mask = cu.asnumpy(dilation(cu.asarray(imagette_mask), str_el))
                        else:
                            str_el = ball(int(config_dict["options"]["dilation"]["str_element_size"]))
                            imagette_mask = dilation(imagette_mask, str_el)

                        imagette *= imagette_mask[None, :, :, :]

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

                    # dilation of mask
                    if json.loads(config_dict["options"]["dilation"]["dilate_mask"].lower()) is True:

                        if cuda is True:
                            str_el = cu.asarray(ball(int(config_dict["options"]["dilation"]["str_element_size"])))
                            imagette_mask = cu.asnumpy(dilation(cu.asarray(imagette_mask), str_el))
                        else:
                            str_el = ball(int(config_dict["options"]["dilation"]["str_element_size"]))
                            imagette_mask = dilation(imagette_mask, str_el)

                        imagette *= imagette_mask

                    concat_image = np.zeros((2, imagette.shape[0], imagette.shape[1], imagette.shape[2]))

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

    def train(viewer, image, mask, region_props_list, labels_list, nn_type, dimension, loss_func, lr, epochs_nb, rot,
              h_flip, v_flip, prob, batch_size, saving_ep, training_name, norm_type, model=None):
        """
        Training of the classification neural network
        @param viewer: Napari viewer instance
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

        global transform, retrain, loaded_network, case

        # Data augmentation
        transforms_list = []
        # This section allows to read as many data augmentation types as we want from the config file:
        for key in config_dict["data augmentation"]:
            if config_dict["data augmentation"][key]["apply"] == "True":
                args_list = list(config_dict["data augmentation"][key].keys())
                args_list.remove("apply")
                s = ""
                for i, arg in enumerate(args_list):
                    s += arg + "=" + config_dict["data augmentation"][key][arg]
                    if i != len(args_list) - 1:
                        s += ","

                transforms_list.append(eval("A." + str(key) + "(" + s + ")"))

        if rot is True:
            transforms_list.append(A.Rotate(-90, 90, p=prob))
        if h_flip is True:
            transforms_list.append(A.HorizontalFlip(p=prob))
        if v_flip is True:
            transforms_list.append(A.VerticalFlip(p=prob))
        transforms_list.append(ToTensorV2())
        transform = A.Compose(transforms_list)

        # List of available network achitectures

        if dimension == "2D":
            nn_dict = {"ResNet18": "resnet18", "ResNet34": "resnet34", "ResNet50": "resnet50", "ResNet101": "resnet101",
                       "ResNet152": "resnet152", "AlexNet": "alexnet", "DenseNet121": "densenet121",
                       "DenseNet161": "densenet161", "DenseNet169": "densenet169", "DenseNet201": "densenet201",
                       "lightNN_2_3": "CNN2D", "lightNN_3_5": "CNN2D", "lightNN_4_5": "CNN2D"}
        else:
            nn_dict = {"lightNN_2_2": "CNN3D", "lightNN_2_4": "CNN3D", "lightNN_2_8": "CNN3D", "lightNN_2_16": "CNN3D",
                       "lightNN_2_32": "CNN3D", "lightNN_2_64": "CNN3D", "lightNN_3_2": "CNN3D", "lightNN_3_4": "CNN3D",
                       "lightNN_3_8": "CNN3D", "lightNN_3_16": "CNN3D", "lightNN_3_32": "CNN3D",
                       "lightNN_3_64": "CNN3D"}
        # Setting of network

        # Concatenation of all the labels lists and conversion to numpy array
        l = []
        # List where empty lists allow to remove the names of the images which have not been labelled
        labels_list_to_clean = labels_list.copy()

        for p in labels_list:
            l += p
        labels_list = np.array(l)

        if retrain is False:
            # 2D case
            if case == "2D":
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

                    if patch_size / (2 ** (depth - 1)) <= kersize:
                        show_info("Patch size is too small for this network")
                    model = CNN2D(max(labels_list) + 1, 4, depth, kersize)

            elif case == "multi3D":
                image = np.transpose(image, (1, 2, 3, 0))
                mask = np.transpose(mask, (1, 2, 0))
                depth = int(nn_type.split("_")[1])
                kersize = int(nn_type.split("_")[2])
                if patch_size / (2 ** (depth - 1)) <= kersize:
                    show_info("Patch size is too small for this network")
                model = CNN3D(max(labels_list) + 1, image.shape[0] + 1, kersize, depth)

            else:
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
                            show_info("Patch size is too small for this network")
                        model = CNN2D(max(labels_list) + 1, image.shape[0] + 1, depth, kersize)

                elif case == "3D":
                    depth = int(nn_type.split("_")[1])
                    kersize = int(nn_type.split("_")[2])
                    if patch_size / (2 ** (depth - 1)) <= kersize:
                        show_info("Patch size is too small for this network")
                    model = CNN3D(max(labels_list) + 1, 2, kersize, depth)


        torch_type = torch.cuda.FloatTensor

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
            optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                         weight_decay=float(config_dict["options"]["optimizer"]["weight_decay"]))
        else:
            optimizer = loaded_network["optimizer_state_dict"]

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=int(config_dict["options"]["learning rate"]["step_size"]),
                                                    gamma=float(config_dict["options"]["learning rate"]["gamma"]))

        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
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
                if case == "2D" or case == "multi2D":
                    # Turn image into 3 channel if it is grayscale
                    if len(image.shape) == 2:
                        image = np.stack((image,) * 3, axis=-1)
                    elif case == "multi2D":
                        image = np.transpose(image, (1, 2, 0))
                    pad_image_list.append(np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                                         (patch_size // 2 + 1, patch_size // 2 + 1), (0, 0)),
                                                 mode="constant"))
                    pad_labels_list.append(np.pad(mask, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant"))

                elif len(image.shape) == 4:
                    image = np.transpose(image, (1, 2, 3, 0))
                    mask = np.transpose(mask, (1, 2, 0))
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
                                     labels_list, torch_type, case, norm_type)
        training_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

        # Optimizer
        model.to("cuda")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print("\t", name)

        # Loss function
        if retrain is False:
            LOSS_LIST = []
        else:
            LOSS_LIST = loaded_network["loss_list"]

        loss = eval("nn." + losses_dict[loss_func] + "().type(torch_type)")

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

                    # Training
                    for local_batch, local_labels in training_loader:
                        # Transfer to GPU
                        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                        out = model(local_batch)
                        total_loss = loss(out, local_labels.type(torch.cuda.FloatTensor))
                        optimizer.zero_grad()
                        total_loss.backward()
                        optimizer.step()
                        # LOSS_LIST.append(total_loss.item())
                        print(total_loss.item())
                        viewer.value.status = "loss = " + str(total_loss.item())
                        # scheduler.step()
                        found = True

                    LOSS_LIST.append(total_loss.item())
                    scheduler.step()
                    if (epoch + 1) % saving_ep == 0:
                        # Learning rate state when saving network
                        print("LR = ", optimizer.param_groups[0]['lr'])

                        d = {"model": model, "optimizer_state_dict": optimizer,
                             "loss": loss, "training_nb": iterations_number, "loss_list": LOSS_LIST,
                             "image_path": image_path_list[0], "labels_path": labels_path_list[0],
                             "patch_size": patch_size, "norm_type": norm_type}
                        if training_name == "":
                            model_path = os.path.join(save_folder, "training_" + str(epoch + 1))
                        else:
                            model_path = os.path.join(save_folder, training_name + "_" + str(epoch + 1))
                        if model_path.endswith(".pt") or model_path.endswith(".pth"):
                            torch.save(d, model_path)
                        else:
                            torch.save(d, model_path + ".pth")

                    yield epoch + 1

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

        return LOSS_LIST

    @magicgui(
        auto_call=True,
        layout='vertical',
        load_data_button=dict(widget_type='PushButton', text='Load data', tooltip='Load the image and the labels'),
        vertical_space1=dict(widget_type='Label', label=' '),
        nn_choice=dict(widget_type='Label', label='NEURAL NETWORK CHOICE'),
        lr=dict(widget_type='LineEdit', label='Learning rate', value=0.01, tooltip='Learning rate'),
        format_button=dict(widget_type='ComboBox', label='Dimension', choices=dims_list, value="2D",
                           tooltip='Choose whether you want to process 2D or 3D images'),
        nn=dict(widget_type='ComboBox', label='Network architecture', choices=networks_list.copy(), value="ResNet18",
                tooltip='All the available network architectures'),
        load_custom_model_button=dict(widget_type='PushButton', text='Load custom model',
                                      tooltip='Load your own NN pretrained or not'),
        loss=dict(widget_type='ComboBox', label='Loss function', choices=losses_list, value="CrossEntropy",
                  tooltip='All the available loss functions'),
        epochs=dict(widget_type='LineEdit', label='Epochs number', value=1000, tooltip='Epochs number'),
        launch_training_button=dict(widget_type='PushButton', text='Launch training', tooltip='Launch training'),
        vertical_space2=dict(widget_type='Label', label=' '),
        DATA_AUGMENTATION_TYPE=dict(widget_type='Label'),
        rotations=dict(widget_type='CheckBox', text='Rotations', tooltip='Random rotations between -90° and 90°'),
        v_flip=dict(widget_type='CheckBox', text='Vertical flip', tooltip='Vertical flip'),
        h_flip=dict(widget_type='CheckBox', text='Horizontal flip', tooltip='Horizontal flip'),
        prob=dict(widget_type='LineEdit', label='Probability', value=0.8, tooltip='Occurrence of a data augmentation'),
        b_size=dict(widget_type='LineEdit', label='Batch Size', value=128, tooltip='Batch Size'),
        data_norm=dict(widget_type='ComboBox', label='Data normalization', choices=data_norm_list,
                       value="min max normalization", tooltip='Type of data normalization'),
        load_config_file_button=dict(widget_type='PushButton', text='Load advanced parameters from config file',
                                tooltip='Load a custom config file containing advanced training parameters from'
                                        ' Svetlana folder'),
        vertical_space3=dict(widget_type='Label', label=' '),
        vertical_space4=dict(widget_type='Label', label=' '),
        SAVING_PARAMETERS=dict(widget_type='Label'),
        saving_ep=dict(widget_type='LineEdit', label='Save training each (epochs)', value=100,
                       tooltip='Each how many epoch the training should be saved'),
        training_name=dict(widget_type='LineEdit', label='Training file name (optional)', tooltip='if not chosen,'
                                                                                                  ' set to "training"'),

    )
    def training_widget(  # label_logo,
            viewer: Viewer,
            load_data_button,
            vertical_space1,
            nn_choice,
            format_button,
            nn,
            load_custom_model_button,
            loss,
            lr,
            epochs,
            b_size,
            data_norm,
            load_config_file_button,
            vertical_space2,
            DATA_AUGMENTATION_TYPE,
            rotations,
            h_flip,
            v_flip,
            prob,
            vertical_space3,
            SAVING_PARAMETERS,
            saving_ep,
            training_name,
            vertical_space4,
            launch_training_button,

    ) -> None:
        # Import when users activate plugin
        # This global instance of the viewer is created to be able to display images from the prediction plugin when
        # it's being opened
        global V
        V = viewer
        return

    @training_widget.format_button.changed.connect
    def set_nets_list(e):
        if e == '3D':
            training_widget.nn.choices = ["lightNN_2_2", "lightNN_2_4", "lightNN_2_8", "lightNN_2_16", "lightNN_2_32",
                                          "lightNN_2_64", "lightNN_3_2", "lightNN_3_4", "lightNN_3_8", "lightNN_3_16",
                                          "lightNN_3_32", "lightNN_3_64"]
            training_widget.nn.value = "lightNN_2_2"
            training_widget.nn.options["choices"].clear()
            training_widget.nn.options["choices"] += ["lightNN_2_2", "lightNN_2_4", "lightNN_2_8", "lightNN_2_16",
                                                      "lightNN_2_32", "lightNN_2_64", "lightNN_3_2", "lightNN_3_4",
                                                      "lightNN_3_8", "lightNN_3_16", "lightNN_3_32", "lightNN_3_64"]
        else:
            training_widget.nn.choices = networks_list
            training_widget.nn.options["choices"].clear()
            training_widget.nn.options["choices"] += networks_list
            training_widget.nn.value = "ResNet18"

    @training_widget.load_data_button.changed.connect
    def _load_data(e: Any):
        """
        Function triggered by the load data button which aims at loading the needed information from the annotation,
        in order to train the NN
        @param e:
        @return:
        """
        training_widget.viewer.value.layers.clear()
        path = QFileDialog.getOpenFileName(None, 'Choose the labels file contained in folder called Svetlana',
                                           options=QFileDialog.DontUseNativeDialog)[0]
        # Necessary to directly load data then, when opening the prediction plugin. The condition to be able to reload
        # the network and the data is that parent_path variable exists
        global parent_path
        parent_path = os.path.split(os.path.split(path)[0])[0]

        try:
            b = torch.load(path)

            global image_path_list, labels_path_list, region_props_list, labels_list, patch_size, image, mask,\
                   config_dict, case

            if "image_path" in b.keys() and "labels_path" in b.keys() and "regionprops" in b.keys() and "labels_list" \
                    in b.keys() and "patch_size" in b.keys():

                image_path_list = b["image_path"]
                labels_path_list = b["labels_path"]
                region_props_list = b["regionprops"]
                labels_list = b["labels_list"]
                patch_size = int(b["patch_size"])

                image = imread(image_path_list[0])
                if len(image.shape) == 2:
                    image = np.stack((image,) * 3, axis=-1)
                mask = imread(labels_path_list[0])
                if len(image.shape) == 4:
                    training_widget.viewer.value.add_image(image, channel_axis=1)
                else:
                    training_widget.viewer.value.add_image(image)
                training_widget.viewer.value.add_labels(mask)
            else:
                show_info("ERROR: The binary file seems not to be correct as it does not contain the right keys")
        except:
            show_info("ERROR: File not recognized by Torch")

        # Load parameters from config file
        try:
            init = os.path.join(os.path.split(os.path.split(np.__file__)[0])[0], "napari_svetlana")
            with open(os.path.join(init, 'Config.json'), 'r') as f:
                config_dict = json.load(f)
        except FileNotFoundError:
            with open(os.path.join(os.getcwd(), 'Config.json'), 'r') as f:
                config_dict = json.load(f)

        # Copy of config file to folder Svetlana
        save_folder = os.path.join(os.path.split(os.path.split(image_path_list[0])[0])[0], "Svetlana")
        if os.path.isdir(save_folder) is False:
            os.mkdir(save_folder)
        import shutil
        if os.path.exists(os.path.join(save_folder, "Config.json")) is False:

            try:
                shutil.copy(os.path.join(init, 'Config.json'),
                        os.path.join(save_folder, "Config.json"))
            except FileNotFoundError:
                shutil.copy(os.path.join(os.getcwd(), 'Config.json'),
                            os.path.join(save_folder, "Config.json"))

        if image.shape[2] <= 3:
            case = "2D"
        elif len(image.shape) == 4:
            case = "multi3D"
        else:
            from .CustomDialog import CustomDialog
            diag = CustomDialog()
            diag.exec()
            case = diag.get_case()
            print(case)

        return

    @training_widget.load_custom_model_button.changed.connect
    def load_custom_model():
        """
        Function triggered by load custom model button to load a pretrained model of the user
        @return:
        """
        path = QFileDialog.getOpenFileName(None, 'Choose the binary file containing the model',
                                           options=QFileDialog.DontUseNativeDialog)[0]
        global loaded_network
        try:
            loaded_network = torch.load(path)
            global model, retrain
            retrain = True
            if "model" in loaded_network.keys():
                model = loaded_network["model"]
                show_info("Model loaded successfully")
            else:
                show_info("ERROR: the file seems not to be correct as it does not contain a key called model")
        except:
            show_info("ERROR: file not recognized by Torch")

    @training_widget.load_config_file_button.changed.connect
    def load_config_file():
        """
        Function to load advanced training parameters from a json file
        @return:
        """
        path = QFileDialog.getOpenFileName(None, 'Choose your custom config file',
                                           options=QFileDialog.DontUseNativeDialog)[0]
        global config_dict
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            show_info("Config file loaded succesfully")
        except:
            show_info("ERROR: Not a correct Json file")

    @training_widget.launch_training_button.changed.connect
    def _launch_training(e: Any):
        """
        Function that calls the training function when launch training button is clicked
        @param e:
        @return:
        """
        if "model" in globals():
            training_worker = thread_worker(train, progress={"total": int(training_widget.epochs.value)}) \
                (training_widget.viewer, image, mask, region_props_list, labels_list, training_widget.nn.value,
                 training_widget.format_button.value, training_widget.loss.value, float(training_widget.lr.value),
                 int(training_widget.epochs.value), training_widget.rotations.value,
                 training_widget.h_flip.value, training_widget.v_flip.value,
                 float(training_widget.prob.value), int(training_widget.b_size.value),
                 int(training_widget.saving_ep.value), str(training_widget.training_name.value),
                 str(training_widget.data_norm.value), model)
            training_worker.returned.connect(plot_loss)
        else:

            training_worker = thread_worker(train, progress={"total": int(training_widget.epochs.value)})(
                training_widget.viewer, image, mask, region_props_list, labels_list,
                training_widget.nn.value, training_widget.format_button.value,
                training_widget.loss.value, float(training_widget.lr.value),
                int(training_widget.epochs.value), training_widget.rotations.value,
                training_widget.h_flip.value, training_widget.v_flip.value,
                float(training_widget.prob.value), int(training_widget.b_size.value),
                int(training_widget.saving_ep.value), str(training_widget.training_name.value),
                str(training_widget.data_norm.value), None)
            training_worker.returned.connect(plot_loss)

        training_worker.start()
        show_info('Training started')

    def plot_loss(loss_list):
        """
        This function gets the loss values list and plots it once the training's thread is done
        @param loss_list:
        @return:
        """
        plt.plot(np.arange(1, len(loss_list) + 1, 1), loss_list)
        plt.title("Training loss")
        plt.xlabel("Epochs number")
        plt.show()

    return training_widget


def Prediction():
    """
    Prediction plugin code
    @return:
    """
    from napari.qt.threading import thread_worker

    def on_pressed(key):
        """
        When a key between 1 and 9 is pressed, it calls the function which sets the label
        @param key: int between 1 and 9
        @return:
        """

        def set_label(viewer):
            """
            Sets a new label when correcting the predicted mask
            @param viewer: Napari viewer instance
            @return:
            """
            global imagette_contours
            if double_click is True and lab != 0:
                imagette_contours[mask == lab] = int(key)
                # Choose whether to label on edges mask or overlay mask
                if prediction_widget.bound.value is True:
                    show_boundaries(True)
                else:
                    viewer.layers.pop()
                    display_result(imagette_contours.astype(np.uint8))

                prediction_widget.viewer.value.layers.selection.active = prediction_widget.viewer.value.layers[
                    "image"]

        return set_label

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

    def draw_uncertainty(prop, imagette_uncertainty, i, list_proba):
        """
        Draw the mask of an object with the colour associated to its predicted class (for 2D images)
        @param compteur: counts the number of objects that belongs to class 1 (int)
        @param prop: region_property of this object
        @param imagette_contours: image of the contours
        @param i: index of the object in the list (int)
        @param list_pred: list of the labels of the classified objects
        @return:
        """

        imagette_uncertainty[prop.coords[:, 0], prop.coords[:, 1]] = list_proba[i].item()

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

    def draw_3D_uncertainty(prop, imagette_uncertainty, i, list_proba):
        """
        Draw the mask of an object with the colour associated to its predicted class (for 2D images)
        @param compteur: counts the number of objects that belongs to class 1 (int)
        @param prop: region_property of this object
        @param imagette_contours: image of the contours
        @param i: index of the object in the list (int)
        @param list_pred: list of the labels of the classified objects
        @return:
        """

        imagette_uncertainty[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = list_proba[i].item() + 1

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
        global imagette_contours, imagette_uncertainty, case

        if case == "2D" or case == "multi2D":
            if case == "multi2D":
                image = np.transpose(image, (1, 2, 0))

            imagette_contours = np.zeros((image.shape[0], image.shape[1]))
            imagette_uncertainty = imagette_contours.copy()
            pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1), (0, 0)), mode="constant")
            pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

            data = PredictionDataset(pad_image, pad_labels, props, patch_size // 2, norm_type, "cuda", config_dict,
                                     case)

        elif case == "multi3D":
            image = np.transpose(image, (1, 2, 3, 0))
            labels = np.transpose(labels, (1, 2, 0))
            imagette_contours = np.zeros((image.shape[3], image.shape[1], image.shape[2]))
            imagette_uncertainty = imagette_contours.copy()
            pad_image = np.pad(image, ((0, 0),
                                       (patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
            pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

            data = PredictionMulti3DDataset(pad_image, pad_labels, props, patch_size // 2, norm_type, "cuda",
                                            config_dict)

        else:
            imagette_contours = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
            imagette_uncertainty = imagette_contours.copy()
            pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
            pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

            data = Prediction3DDataset(pad_image, pad_labels, props, patch_size // 2, norm_type, "cuda", config_dict)
        prediction_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

        global list_pred
        list_pred = []
        list_proba = []
        for i, local_batch in enumerate(prediction_loader):
            with torch.no_grad():
                out = model(local_batch)
                if out.dim() == 1:
                    out = out[:, None]
                proba, index = torch.max(out, 1)
                list_pred += index
                list_proba += proba
                yield i + 1

        show_info("Prediction of patches done, please wait while the result image is being generated...")
        if len(labels.shape) == 2:
            compteur = Parallel(n_jobs=-1, require="sharedmem")(
                delayed(draw_predicted_contour)(compteur, prop, imagette_contours, i, list_pred)
                for i, prop in enumerate(props))
            # if chekbox is True, ccomputes the probabilities mask of confidence
            if prediction_widget.confidence_button.value is True:
                Parallel(n_jobs=-1, require="sharedmem")(
                    delayed(draw_uncertainty)(prop, imagette_uncertainty, i, list_proba) for i, prop in enumerate(props))

        else:
            compteur = Parallel(n_jobs=-1, require="sharedmem")(
                delayed(draw_3d_prediction)(compteur, prop, imagette_contours, i, list_pred)
                for i, prop in enumerate(props))
            # if chekbox is True, ccomputes the probabilities mask of confidence
            if prediction_widget.confidence_button.value is True:
                Parallel(n_jobs=-1, require="sharedmem")(
                    delayed(draw_3D_uncertainty)(prop, imagette_uncertainty, i, list_proba) for i, prop in enumerate(props))

        if prediction_widget.confidence_button.value is True:
            imagette_uncertainty = ((imagette_uncertainty - imagette_uncertainty.min()) / (
                                    imagette_uncertainty.max() - imagette_uncertainty.min()) * 255).astype("uint8")
            if case == "2D" or case == "multi2D":
                imagette_uncertainty = cv2.applyColorMap(imagette_uncertainty, cv2.COLORMAP_JET)
            else:
                imagette_uncertainty3d = np.zeros((*imagette_uncertainty.shape, 3))
                for i in range(imagette_uncertainty.shape[0]):
                    imagette_uncertainty3d[i, :, :, :] = cv2.applyColorMap(imagette_uncertainty[i, :, :],
                                                                           cv2.COLORMAP_JET)
                imagette_uncertainty = imagette_uncertainty3d.copy().astype("uint8")

            imagette_uncertainty[mask == 0] = 0

        # Deletion of the old mask
        prediction_widget.viewer.value.layers.pop()

        stop = time.time()
        print("temps de traitement", stop - start)
        show_info(str(np.sum(compteur)) + " objects remaining over " + str(len(props)))
        print(str(np.sum(compteur)) + " objects remaining over " + str(len(props)))

        # Save the result automatically
        res_name = "prediction_" + os.path.split(image_path_list[int(prediction_widget.image_index_button.value) - 1])[
            1]
        conf_name = "confidence_" + os.path.split(image_path_list[int(prediction_widget.image_index_button.value) - 1])[
            1]
        imsave(os.path.join(res_folder, res_name), imagette_contours.astype(np.uint8))
        if prediction_widget.confidence_button.value is True:
            imsave(os.path.join(conf_folder, conf_name), imagette_uncertainty)

        return imagette_contours.astype(np.uint8)

    def predict_batch(image_path_list, mask_path_list, patch_size, batch_size):
        """
        Prediction of the class of each patch extracted from the great mask
        @param image: raw image
        @param labels: segmentation mask
        @param patch_size: size of the patches to be classified (int)
        @param batch_size: batch size for the NN (int)
        @return:
        """
        df_path = os.path.join(os.path.split(images_folder)[0], "Svetlana", "prediction_regionprops")
        df_list = []
        for ind in range(0, len(image_path_list)):
            image = imread(image_path_list[ind])
            labels = imread(mask_path_list[ind])
            props = regionprops(labels)

            import time
            start = time.time()
            compteur = 0
            global imagette_contours, imagette_uncertainty

            if len(image.shape) == 2:
                image = np.stack((image,) * 3, axis=-1)

            if case == "2D" or case == "multi2D":
                if case == "multi2D":
                    image = np.transpose(image, (1, 2, 0))

                imagette_contours = np.zeros((image.shape[0], image.shape[1]))
                imagette_uncertainty = imagette_contours.copy()

                pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                           (patch_size // 2 + 1, patch_size // 2 + 1), (0, 0)), mode="constant")
                pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                             (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

                data = PredictionDataset(pad_image, pad_labels, props, patch_size // 2, norm_type, "cuda", config_dict,
                                     case)

            elif case == "multi3D":
                image = np.transpose(image, (1, 2, 3, 0))
                labels = np.transpose(labels, (1, 2, 0))
                imagette_contours = np.zeros((image.shape[3], image.shape[1], image.shape[2]))
                imagette_uncertainty = imagette_contours.copy()
                pad_image = np.pad(image, ((0, 0),
                                           (patch_size // 2 + 1, patch_size // 2 + 1),
                                           (patch_size // 2 + 1, patch_size // 2 + 1),
                                           (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
                pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                             (patch_size // 2 + 1, patch_size // 2 + 1),
                                             (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

                data = PredictionMulti3DDataset(pad_image, pad_labels, props, patch_size // 2, norm_type, "cuda",
                                                config_dict)

            else:
                imagette_contours = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
                imagette_uncertainty = imagette_contours.copy()
                pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                           (patch_size // 2 + 1, patch_size // 2 + 1),
                                           (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
                pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                             (patch_size // 2 + 1, patch_size // 2 + 1),
                                             (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

                data = Prediction3DDataset(pad_image, pad_labels, props, patch_size // 2, norm_type, "cuda",
                                           config_dict)
            prediction_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)

            global list_pred
            list_pred = []
            list_proba = []
            for i, local_batch in enumerate(prediction_loader):
                with torch.no_grad():
                    out = model(local_batch)
                    if out.dim() == 1:
                        out = out[:, None]
                    proba, index = torch.max(out, 1)
                    list_pred += index
                    list_proba += proba
                    yield i + 1

            show_info("Prediction of patches done, please wait while the result image is being generated...")
            if len(labels.shape) == 2:
                compteur = Parallel(n_jobs=-1, require="sharedmem")(
                    delayed(draw_predicted_contour)(compteur, prop, imagette_contours, i, list_pred)
                    for i, prop in enumerate(props))
                # if chekbox is True, ccomputes the probabilities mask of confidence
                if prediction_widget.confidence_button.value is True:
                    Parallel(n_jobs=-1, require="sharedmem")(
                        delayed(draw_uncertainty)(prop, imagette_uncertainty, i, list_proba) for i, prop in
                        enumerate(props))
            else:
                compteur = Parallel(n_jobs=-1, require="sharedmem")(
                    delayed(draw_3d_prediction)(compteur, prop, imagette_contours, i, list_pred)
                    for i, prop in enumerate(props))
                # if chekbox is True, ccomputes the probabilities mask of confidence
                if prediction_widget.confidence_button.value is True:
                    Parallel(n_jobs=-1, require="sharedmem")(
                        delayed(draw_3D_uncertainty)(prop, imagette_uncertainty, i, list_proba) for i, prop in
                        enumerate(props))

            if prediction_widget.confidence_button.value is True:
                imagette_uncertainty = ((imagette_uncertainty - imagette_uncertainty.min()) / (
                        imagette_uncertainty.max() - imagette_uncertainty.min()) * 255).astype("uint8")
                if case == "2D" or case == "multi2D":
                    imagette_uncertainty = cv2.applyColorMap(imagette_uncertainty, cv2.COLORMAP_JET)
                else:
                    imagette_uncertainty3d = np.zeros((*imagette_uncertainty.shape, 3))
                    for i in range(imagette_uncertainty.shape[0]):
                        imagette_uncertainty3d[i, :, :, :] = cv2.applyColorMap(imagette_uncertainty[i, :, :],
                                                                               cv2.COLORMAP_JET)
                    imagette_uncertainty = imagette_uncertainty3d.copy().astype("uint8")

                imagette_uncertainty[mask == 0] = 0

            stop = time.time()
            print("temps de traitement", stop - start)
            show_info(str(np.sum(compteur)) + " objects remaining over " + str(len(props)))
            print(str(np.sum(compteur)) + " objects remaining over " + str(len(props)))

            # Save the result automatically
            res_name = "prediction_" + os.path.split(image_path_list[ind])[1]
            conf_name = "confidence_" + os.path.split(image_path_list[ind])[1]

            imsave(os.path.join(res_folder, res_name), imagette_contours.astype(np.uint8))
            if prediction_widget.confidence_button.value is True:
                imsave(os.path.join(conf_folder, conf_name), imagette_uncertainty)

            show_info("prediction of image " + os.path.split(image_path_list[ind])[1] + " done")

            df_list.append(save_regionprops_all(ind, props, list_pred))

        d = pd.concat(df_list)
        d.to_excel(df_path + ".xlsx", engine="xlsxwriter", index=False)
        show_info("ROI properties saved for all images")

    def show_cam(image, labels, props, patch_size, lab):

        if case == "2D" or case == "multi2D":
            if case == "multi2D":
                image = np.transpose(image, (1, 2, 0))

            pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1), (0, 0)), mode="constant")
            pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

            for i, reg in enumerate(props):
                if reg.label == lab:
                    ind = i
            input_tensor = PredictionDataset(pad_image, pad_labels, props, patch_size // 2, norm_type, "cuda", config_dict,
                                     case).__getitem__(ind)[None, :]
            model.eval()

            import torchvision
            if (type(model) == torchvision.models.resnet.ResNet) is True:
                target_layers = [model.layer4[-1]]
            elif (str(type(model)) == '<class \'torchvision.models.alexnet.AlexNet\'>') is True:
                target_layers = [model.features[-2]]
            elif (type(model) == torchvision.models.densenet.DenseNet) is True:
                target_layers = [model.features[-2]]
            else:
                target_layers = [model.cnn_layers]

            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

            with torch.no_grad():
                out = model(input_tensor)
                if out.dim() == 1:
                    out = out[:, None]
                proba, index = torch.max(out, 1)

            targets = [ClassifierOutputTarget(index[0].item())]
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)

            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            np_arr = np.zeros((input_tensor.shape[2], input_tensor.shape[3], 3))
            for j in range(0, 3):
                np_arr[:, :, j] = input_tensor[0, 0, :, :].cpu().detach().numpy().copy()

            np_arr = (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min())
            grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())

            visualization = show_cam_on_image(np_arr.astype(np.float32), grayscale_cam, use_rgb=True)

            # Display result at the right coordinates
            x_corner = (int(props[ind].centroid[0]) + patch_size // 2 + 1) - patch_size // 2 - (patch_size // 2 + 1)
            y_corner = (int(props[ind].centroid[1]) + patch_size // 2 + 1) - patch_size // 2 - (patch_size // 2 + 1)

            prediction_widget.viewer.value.add_image(visualization, name="%.1f" % (proba * 100) + "% G-CAM",
                                                     translate=(x_corner, y_corner))

            # Guided backpropagation :
            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
            gb = gb_model(input_tensor, target_category=None)

            import cv2
            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = deprocess_image(cam_mask * gb[:, :, :3])
            cam_gb = (cam_gb - cam_gb.min()) / (cam_gb.max() - cam_gb.min())
            cam_gb = show_cam_on_image(np_arr.astype(np.float32), cam_gb, use_rgb=True)

            prediction_widget.viewer.value.add_image(cam_gb, name="Guided backpropagation GRAD-CAM",
                                                     translate=(x_corner, y_corner))

            # Stay focus on image layer to keep annotating
            prediction_widget.viewer.value.layers.selection.active = prediction_widget.viewer.value.layers[
                "image"]
        elif case == "3D":
            pad_image = np.pad(image, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
            pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

            for i, reg in enumerate(props):
                if reg.label == lab:
                    ind = i

            input_tensor = Prediction3DDataset(pad_image, pad_labels, props, patch_size // 2, norm_type, "cuda",
                                               config_dict).__getitem__(ind)[None, :]

            model.eval()
            import torchvision
            if (type(model) == torchvision.models.resnet.ResNet) is True:
                target_layers = [model.layer4[-1]]
            elif (str(type(model)) == '<class \'torchvision.models.alexnet.AlexNet\'>') is True:
                target_layers = [model.features[-2]]
            elif (type(model) == torchvision.models.densenet.DenseNet) is True:
                target_layers = [model.features[-2]]
            else:
                target_layers = [model.cnn_layers]

            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

            with torch.no_grad():
                out = model(input_tensor)
                if out.dim() == 1:
                    out = out[:, None]
                proba, index = torch.max(out, 1)

            targets = [ClassifierOutputTarget(index[0].item())]
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)

            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            c = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min()) * 255

            # Display result at the right coordinates
            x_corner = (int(props[ind].centroid[0]) + patch_size // 2 + 1) - patch_size // 2 - (patch_size // 2 + 1)
            y_corner = (int(props[ind].centroid[1]) + patch_size // 2 + 1) - patch_size // 2 - (patch_size // 2 + 1)
            z_corner = (int(props[ind].centroid[2]) + patch_size // 2 + 1) - patch_size // 2 - (patch_size // 2 + 1)

            prediction_widget.viewer.value.add_image(c, colormap="turbo", opacity=0.7,
                                                     name="%.1f" % (proba * 100) + "% G-CAM",
                                                     translate=(x_corner, y_corner, z_corner))
            # Stay focus on image layer to keep annotating
            prediction_widget.viewer.value.layers.selection.active = prediction_widget.viewer.value.layers[
                "image"]
        else:
            image = np.transpose(image, (1, 2, 3, 0))
            labels = np.transpose(labels, (1, 2, 0))
            pad_image = np.pad(image, ((0, 0),
                                       (patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1),
                                       (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
            pad_labels = np.pad(labels, ((patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1),
                                         (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")

            for i, reg in enumerate(props):
                if reg.label == lab:
                    ind = i

            input_tensor = PredictionMulti3DDataset(pad_image, pad_labels, props, patch_size // 2, norm_type, "cuda",
                                                    config_dict).__getitem__(ind)[None, :]

            model.eval()
            import torchvision
            if (type(model) == torchvision.models.resnet.ResNet) is True:
                target_layers = [model.layer4[-1]]
            elif (str(type(model)) == '<class \'torchvision.models.alexnet.AlexNet\'>') is True:
                target_layers = [model.features[-2]]
            elif (type(model) == torchvision.models.densenet.DenseNet) is True:
                target_layers = [model.features[-2]]
            else:
                target_layers = [model.cnn_layers]

            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

            with torch.no_grad():
                out = model(input_tensor)
                if out.dim() == 1:
                    out = out[:, None]
                proba, index = torch.max(out, 1)

            targets = [ClassifierOutputTarget(index[0].item())]
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)

            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            c = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min()) * 255
            # Display result at the right coordinates
            x_corner = (int(props[ind].centroid[0]) + patch_size // 2 + 1) - patch_size // 2 - (patch_size // 2 + 1)
            y_corner = (int(props[ind].centroid[1]) + patch_size // 2 + 1) - patch_size // 2 - (patch_size // 2 + 1)
            z_corner = (int(props[ind].centroid[2]) + patch_size // 2 + 1) - patch_size // 2 - (patch_size // 2 + 1)

            prediction_widget.viewer.value.add_image(c, colormap="turbo", opacity=0.7,
                                                     name="%.1f" % (proba * 100) + "% G-CAM",
                                                     translate=(x_corner, y_corner, z_corner))
            # Stay focus on image layer to keep annotating
            prediction_widget.viewer.value.layers.selection.active = prediction_widget.viewer.value.layers[
                "image"]

    def load_data_after_training():
        """
        The aim of this function is to load the NN and the data directly in the prediction plugin without having to use
         load data and NN buttons
        @return:
        """
        # If parent path exists, then an annotation, or a training have been performed juste before in the same
        # Svetlana's instance
        if "parent_path" in globals():

            V.layers.clear()

            p = os.path.join(parent_path, "Svetlana")
            onlyfiles = [os.path.join(p, f) for f in os.listdir(p) if
                         os.path.isfile(os.path.join(p, f)) and os.path.join(p, f).endswith(".pth")]
            if len(onlyfiles) > 0:
                path = max(onlyfiles, key=os.path.getctime)

                # Loading of last trained NN
                try:
                    b = torch.load(path)

                    global model, patch_size, norm_type

                    if "model" and "patch_size" and "norm_type" in b.keys():
                        model = b["model"].to("cuda")
                        patch_size = b["patch_size"]
                        norm_type = b["norm_type"]
                        model.eval()
                        show_info("NN loaded successfully")
                    else:
                        show_info("ERROR: the file seems not be correct as it does not contain the right keys")
                except:
                    show_info("ERROR: file not recognized by Torch")

                # Loading the data fot he prediction
                path = parent_path
                global config_dict
                # Load parameters from config file
                with open(os.path.join(path, "Svetlana", 'Config.json'), 'r') as f:
                    config_dict = json.load(f)

                # Result folder
                global res_folder, conf_folder
                res_folder = os.path.join(path, "Predictions")
                if os.path.isdir(res_folder) is False:
                    os.mkdir(res_folder)

                conf_folder = os.path.join(path, "Confidence")
                if os.path.isdir(conf_folder) is False:
                    os.mkdir(conf_folder)

                global images_folder, masks_folder
                images_folder = os.path.join(path, "Images")
                masks_folder = os.path.join(path, "Masks")

                if os.path.isdir(images_folder) is True and os.path.isdir(masks_folder) is True:
                    global image_path_list, mask_path_list
                    image_path_list = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder)])
                    mask_path_list = sorted([os.path.join(masks_folder, f) for f in os.listdir(masks_folder)])

                    global image, mask

                    image = imread(image_path_list[0])
                    if len(image.shape) == 2:
                        image = np.stack((image,) * 3, axis=-1)
                    mask = imread(mask_path_list[0])

                    # If the image is 3D multichannel, it is splitted into several images
                    if len(image.shape) == 4:
                        V.add_image(image, channel_axis=1, name="image")
                    else:
                        V.add_image(image, name="image")
                    V.add_labels(mask, name="mask")

                    # Must be called at the end of loading data so the layer for labeling bay double clicking can be defined as
                    # the layer named Image
                    # prediction_widget()
                else:
                    show_info("ERROR: The folder should contain two folders called Images and Masks")

    @magicgui(
        auto_call=False,
        call_button=False,
        layout='vertical',
        batch_size=dict(widget_type='LineEdit', label='Batch size', value=100, tooltip='Batch size'),
        load_network_button=dict(widget_type='PushButton', text='Load network', tooltip='Load weights of the NN'),
        load_data_button=dict(widget_type='PushButton', text='Load data', tooltip='Load images to process'),
        vertical_space1=dict(widget_type='Label', label=' '),
        vertical_space2=dict(widget_type='Label', label=' '),
        vertical_space3=dict(widget_type='Label', label=' '),
        vertical_space4=dict(widget_type='Label', label=' '),
        network_running=dict(widget_type='Label', label='NETWORK RUNNING'),
        image_choice=dict(widget_type='Label', label="IMAGE CHOICE"),
        classified_mask=dict(widget_type='Label', label="CLASSIFIED MASK"),
        image_index_button=dict(widget_type='LineEdit', label='Image index', value=1,
                                tooltip='Image index in the batch'),
        previous_button=dict(widget_type='PushButton', text='Previous image',
                             tooltip='Previous image'),
        next_button=dict(widget_type='PushButton', text='Next image',
                         tooltip='Next image'),
        launch_prediction_button=dict(widget_type='PushButton', text='Predict this image',
                                      tooltip='Predict the current image'),
        launch_batch_prediction_button=dict(widget_type='PushButton', text='Predict whole batch',
                                            tooltip='Predict the whole batch and save the result'),
        confidence_button=dict(widget_type='CheckBox', text='Compute confidence mask', tooltip='Calculate a heat map'
                                                                                               ' showing where the '
                                                                                               'network has lacked'
                                                                                               ' confidence in its'
                                                                                               ' prediction'),
        bound=dict(widget_type='CheckBox', text='Show boundaries only', tooltip='Show boundaries only'),
        edges_thickness=dict(widget_type='LineEdit', label='Edges thickness', value=7,
                             tooltip='Edges thickness'),
        generate_im_labs_button=dict(widget_type='PushButton', text='Save masks of labels', tooltip='Save one '
                                                                                                    'per attributed label'),
        save_regionprops_button=dict(widget_type='PushButton', text='Save objects statistics', tooltip='Save the '
                                                                                                       'properties of the annotated objects in a binary file, loadable using torch.load'),
        click_annotate=dict(widget_type='CheckBox', text='Click to change label', tooltip='Click to change label'),
        show_heatmap=dict(widget_type='CheckBox', text='Compute Grad-CAM',
                          tooltip='Click on a cell to show the heat map'),
    )
    def prediction_widget(  # label_logo,
            viewer: Viewer,
            load_network_button,
            load_data_button,
            vertical_space1,
            image_choice,
            previous_button,
            image_index_button,
            next_button,
            vertical_space2,
            network_running,
            batch_size,
            launch_prediction_button,
            launch_batch_prediction_button,
            confidence_button,
            vertical_space3,
            classified_mask,
            show_heatmap,
            click_annotate,
            bound,
            edges_thickness,
            vertical_space4,
            save_regionprops_button,
            generate_im_labs_button

    ) -> None:
        # Import when users activate plugin
        # We generate the functions to add a label when a key i pressed
        for i in range(1, 10):
            viewer.bind_key(str(i), on_pressed(i), overwrite=True)

        if len(viewer.layers) > 0:
            global layer, double_click

            layer = prediction_widget.viewer.value.layers["image"]

            @layer.mouse_double_click_callbacks.append
            def label_clicking(layer, event):
                """
                When click to annotate option is activated, retrieves the coordinate of the clicked object to give him a
                label
                @param layer:
                @param event: Qt click event
                @return:
                """
                # Make sure there is only one layer dedicated to double clicking feature
                while len(layer.mouse_double_click_callbacks) > 1:
                    layer.mouse_double_click_callbacks.pop()
                global lab
                if double_click is True:
                    if case == "2D":
                        lab = mask[int(event.position[0]), int(event.position[1])]
                    elif case == "multi2D":
                        lab = mask[int(event.position[1]), int(event.position[2])]
                    elif case == "3D":
                        lab = mask[int(event.position[0]), int(event.position[1]), int(event.position[2])]
                    else:
                        lab = mask[int(event.position[0]), int(event.position[1]), int(event.position[2])]

                    if lab != 0:
                        show_info("Choose a label for that object")
                    else:
                        show_info("Not an object")

                if heatmap is True:
                    if case == "2D":
                        lab = mask[int(event.position[0]), int(event.position[1])]
                        show_cam(image, mask, props, patch_size, lab)
                    elif case == "multi2D":
                        lab = mask[int(event.position[1]), int(event.position[2])]
                        show_cam(image, mask, props, patch_size, lab)
                    elif case == "3D" or case == "multi3D":
                        lab = mask[int(event.position[0]), int(event.position[1]), int(event.position[2])]
                        show_cam(image, mask, props, patch_size, lab)

    @prediction_widget.show_heatmap.changed.connect
    def click_to_show_heatmap(e: Any):
        global heatmap
        if e is True:
            heatmap = True
            # select the image so the user can click on it
            prediction_widget.viewer.value.layers.selection.active = prediction_widget.viewer.value.layers[
                "image"]
        else:
            heatmap = False
        return heatmap

    @prediction_widget.click_annotate.changed.connect
    def click_to_annotate(e: Any):
        global double_click
        if e is True:
            double_click = True
            # select the image so the user can click on it
            prediction_widget.viewer.value.layers.selection.active = prediction_widget.viewer.value.layers[
                "image"]
        else:
            double_click = False
        return double_click

    @prediction_widget.load_network_button.changed.connect
    def load_network(e: Any):
        """
        Function triggered by the load data button to load the weights of the neural network
        @param e:
        @return:
        """
        # Removal of the remaining images of the previous widgets
        prediction_widget.viewer.value.layers.clear()
        prediction_widget()

        path = QFileDialog.getOpenFileName(None, 'Choose the binary file containing the model',
                                           options=QFileDialog.DontUseNativeDialog)[0]
        """
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        """
        try:
            b = torch.load(path)

            global model, patch_size, norm_type

            if "model" and "patch_size" and "norm_type" in b.keys():
                model = b["model"].to("cuda")
                patch_size = b["patch_size"]
                norm_type = b["norm_type"]
                model.eval()
                show_info("NN loaded successfully")
            else:
                show_info("ERROR: the file seems not be correct as it does not contain the right keys")
        except:
            show_info("ERROR: file not recognized by Torch")

    @prediction_widget.load_data_button.changed.connect
    def _load_data(e: Any):
        """
        Function triggered by the load data button to load the weights of the neural network
        @param e:
        @return:
        """
        # Removal of the remaining images of the previous widgets
        prediction_widget.viewer.value.layers.clear()

        path = QFileDialog.getExistingDirectory(None, 'Choose the parent folder which contains folders Images '
                                                      'and Masks', options=QFileDialog.DontUseNativeDialog)

        global config_dict
        # Load parameters from config file
        with open(os.path.join(path, "Svetlana", 'Config.json'), 'r') as f:
            config_dict = json.load(f)

        # Result folder
        global res_folder, conf_folder
        res_folder = os.path.join(path, "Predictions")
        if os.path.isdir(res_folder) is False:
            os.mkdir(res_folder)

        conf_folder = os.path.join(path, "Confidence")
        if os.path.isdir(conf_folder) is False:
            os.mkdir(conf_folder)

        global images_folder, masks_folder
        images_folder = os.path.join(path, "Images")
        masks_folder = os.path.join(path, "Masks")

        if os.path.isdir(images_folder) is True and os.path.isdir(masks_folder) is True:
            global image_path_list, mask_path_list
            image_path_list = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder)])
            mask_path_list = sorted([os.path.join(masks_folder, f) for f in os.listdir(masks_folder)])

            global image, mask

            image = imread(image_path_list[0])
            if len(image.shape) == 2:
                image = np.stack((image,) * 3, axis=-1)
            mask = imread(mask_path_list[0])
            # If the image is 3D multichannel, it is splitted into several images
            if len(image.shape) == 4:
                prediction_widget.viewer.value.add_image(image, channel_axis=1, name="image")
            else:
                prediction_widget.viewer.value.add_image(image, name="image")
            prediction_widget.viewer.value.add_labels(mask, name="mask")

            # Set the format of the image for the prediction (useful pour the click to change label)
            global case, zoom_factor

            if image.shape[2] <= 3:
                case = "2D"
            elif len(image.shape) == 4:
                case = "multi3D"
            else:
                from .CustomDialog import CustomDialog
                diag = CustomDialog()
                diag.exec()
                case = diag.get_case()
                print(case)

            # Must be called at the end of loading data so the layer for labeling bay double clicking can be defined as
            # the layer named Image
            prediction_widget()
        else:
            show_info("ERROR: The folder should contain two folders called Images and Masks")

        return

    @prediction_widget.image_index_button.changed.connect
    def set_image_index(e: Any):
        """
        Set image index instead of using previous/next
        """
        if int(e) > len(image_path_list):
            prediction_widget.image_index_button.value = len(image_path_list)
        elif int(e) < 1:
            prediction_widget.image_index_button.value = 1

        global image, mask
        image = imread(image_path_list[int(prediction_widget.image_index_button.value) - 1])
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        mask = imread(mask_path_list[int(prediction_widget.image_index_button.value) - 1])
        prediction_widget.viewer.value.layers.clear()
        # If the image is 3D multichannel, it is splitted into several images
        if len(image.shape) == 4:
            prediction_widget.viewer.value.add_image(image, channel_axis=1, name="image")
        else:
            prediction_widget.viewer.value.add_image(image, name="image")
        prediction_widget.viewer.value.add_labels(mask, name="mask")

        # Must be called at the end of loading data so the layer for labeling bay double clicking can be defined as
        # the layer named Image
        prediction_widget()

    @prediction_widget.previous_button.changed.connect
    def load_previous_image(e: Any):
        """
        Turn to previous image
        """
        if int(prediction_widget.image_index_button.value) > 1:
            prediction_widget.image_index_button.value = int(prediction_widget.image_index_button.value) - 1
            # Must be called at the end of loading data so the layer for labeling bay double clicking can be defined as
            # the layer named Image
            prediction_widget()
        else:
            show_info("No previous image")

    @prediction_widget.next_button.changed.connect
    def load_next_image(e: Any):
        """
        Turn to next image
        """
        if int(prediction_widget.image_index_button.value) < len(image_path_list):
            prediction_widget.image_index_button.value = int(prediction_widget.image_index_button.value) + 1
            # Must be called at the end of loading data so the layer for labeling bay double clicking can be defined as
            # the layer named Image
            prediction_widget()
        else:
            show_info("No more images")

    def display_result(image):
        """
        Displays the classified mask once the neural network prediction is over
        @param image: classified mask to display as an overlay
        @return:
        """
        prediction_widget.viewer.value.add_labels(image)
        # Call the function to define image as the layer for double clicking feature
        prediction_widget()
        # Chose of the colours
        prediction_widget.viewer.value.layers[-1].name = "Classified labels"
        if len(np.unique(prediction_widget.viewer.value.layers["Classified labels"].data)) == 3:
            prediction_widget.viewer.value.layers["Classified labels"].color = {1: "green", 2: "red"}

    @prediction_widget.launch_prediction_button.changed.connect
    def _launch_prediction(e: Any):
        """
        Calls the prediciton when the button is triggered
        @param e:
        @return:
        """
        global props
        props = regionprops(mask)
        prediction_worker = thread_worker(predict, progress={
            "total": int(np.ceil(len(props) / int(prediction_widget.batch_size.value)))}) \
            (image, mask, props, patch_size, int(prediction_widget.batch_size.value))
        # Addition of the new labels
        prediction_worker.returned.connect(display_result)

        prediction_worker.start()
        show_info('Prediction started')

    @prediction_widget.launch_batch_prediction_button.changed.connect
    def _launch_batch_prediction(e: Any):
        """
        Calls the prediciton when the button is triggered
        @param e:
        @return:
        """

        # estimation of total number of patches to compute through the batch
        props = []
        for m in mask_path_list:
            mask = imread(m)
            props += regionprops(mask)

        prediction_worker = thread_worker(predict_batch, progress={
            "total": int(np.ceil(len(props) / int(prediction_widget.batch_size.value)))}) \
            (image_path_list, mask_path_list, patch_size, int(prediction_widget.batch_size.value))

        prediction_worker.start()
        show_info("Prediction started")

    @prediction_widget.bound.changed.connect
    def show_boundaries(e: Any):
        """
        Only show the edges of the predicted mask instead of an overlay (only for 2D)
        @param e:
        @param size: edges thickness
        @return:
        """
        global eroded_labels, thickness
        if "edge_im" not in globals():
            # computation of the cells segmentation edges
            eroded_contours = cv2.erode(np.uint16(mask), np.ones((int(thickness), int(thickness)), np.uint8))
            eroded_labels = mask - eroded_contours

        if e is True:
            # Removing the inside of the cells in the binary result using the edges mask computed just before
            edge_im = imagette_contours.copy().astype(np.uint8)
            edge_im[eroded_labels == 0] = 0
            pyramidal_edge_im = [edge_im]
            for i in range(1, 6):
                pyramidal_edge_im.append(cv2.resize(edge_im, (edge_im.shape[0] // 2 ** i,
                                                              edge_im.shape[1] // 2 ** i)))
            prediction_widget.viewer.value.layers.pop()
            prediction_widget.viewer.value.add_labels(pyramidal_edge_im, name="Classified labels")
            if len(np.unique(prediction_widget.viewer.value.layers[1].data)) == 3:
                prediction_widget.viewer.value.layers["Classified labels"].color = {1: "green", 2: "red"}
        else:
            pyramidal_imagette_contours = [imagette_contours.astype(np.uint8)]
            for i in range(1, 6):
                pyramidal_imagette_contours.append(
                    cv2.resize(imagette_contours.astype(np.uint8), (imagette_contours.shape[0] // 2 ** i,
                                                                    imagette_contours.shape[
                                                                        1] // 2 ** i)))
            prediction_widget.viewer.value.layers.pop()
            prediction_widget.viewer.value.add_labels(pyramidal_imagette_contours, name="Classified labels")
            if len(np.unique(prediction_widget.viewer.value.layers["Classified labels"].data)) == 3:
                prediction_widget.viewer.value.layers["Classified labels"].color = {1: "green", 2: "red"}

        # make image active so it can be labelled
        prediction_widget.viewer.value.layers.selection.active = prediction_widget.viewer.value.layers["image"]

    @prediction_widget.edges_thickness.changed.connect
    def set_edges_thickness(e: Any):
        """
        Function which changes the edges thickness in the mask
        @param e:
        @return:
        """
        global thickness

        if prediction_widget.bound.value is True:
            thickness = int(e)
            show_boundaries(True)

    @prediction_widget.save_regionprops_button.changed.connect
    def save_regionprops():
        """
        Saves the properties of the labelled connected components of the image in a xlsx file
        @return:
        """

        path = os.path.join(os.path.split(images_folder)[0], "Svetlana", "prediction_regionprops")
        if os.path.isdir(os.path.split(path)[0]) is False:
            os.mkdir(os.path.split(path)[0])
        props_list = []
        # Image name added to the list
        props_list.append(image_path_list[int(prediction_widget.image_index_button.value) - 1])
        if len(mask.shape) == 3:
            for i, prop in enumerate(props):
                props_list.append({"position": prop.label, "coords": prop.coords, "centroid": prop.centroid,
                                   "area": prop.area, "label": int(list_pred[i].item())})
        else:
            for i, prop in enumerate(props):
                props_list.append({"position": prop.label, "coords": prop.coords, "centroid": prop.centroid,
                                   "eccentricity": prop.eccentricity, "area": prop.area, "perimeter": prop.perimeter,
                                   "label": int(list_pred[i].item()) + 1})
        df = pd.DataFrame.from_dict(props_list[1:])
        df.insert(loc=0,
                  column='image_name',
                  value=[props_list[0]] * len(props_list[1:]))
        df.to_excel(path + ".xlsx", engine="xlsxwriter", index=False)

        show_info("ROI properties saved for this image")

    def save_regionprops_all(index, props, list_pred):
        """
        Saves the properties of the labelled connected components of the whole batch in a xlsx file
        @return:
        """

        path = os.path.join(os.path.split(images_folder)[0], "Svetlana", "prediction_regionprops")
        if os.path.isdir(os.path.split(path)[0]) is False:
            os.mkdir(os.path.split(path)[0])
        props_list = []
        # Image name added to the list
        props_list.append(image_path_list[index])
        if len(mask.shape) == 3:
            for i, prop in enumerate(props):
                props_list.append({"position": prop.label, "coords": prop.coords, "centroid": prop.centroid,
                                   "area": prop.area, "label": int(list_pred[i].item())})
        else:
            for i, prop in enumerate(props):
                props_list.append({"position": prop.label, "coords": prop.coords, "centroid": prop.centroid,
                                   "eccentricity": prop.eccentricity, "area": prop.area, "perimeter": prop.perimeter,
                                   "label": int(list_pred[i].item()) + 1})
        df = pd.DataFrame.from_dict(props_list[1:])
        df.insert(loc=0,
                  column='image_name',
                  value=[props_list[0]] * len(props_list[1:]))
        return df

    @prediction_widget.generate_im_labs_button.changed.connect
    def generate_im_labels():
        """
        Saves a mask for each class
        @return:
        """
        im_labs_list = []
        # We create as many images as labels
        for i in range(0, max(list_pred).item() + 1):
            im_labs_list.append(np.zeros_like(mask).astype(np.uint16))

        if len(mask.shape) == 3:
            for i, prop in enumerate(props):
                im_labs_list[list_pred[i]][prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = prop.label
        else:
            for i, prop in enumerate(props):
                im_labs_list[list_pred[i]][prop.coords[:, 0], prop.coords[:, 1]] = prop.label

        for i, im in enumerate(im_labs_list):
            imsave(os.path.splitext(mask_path_list[int(prediction_widget.image_index_button.value) - 1])[0] +
                   "_label" + str(i + 1) + ".tif", im)

    # Function called when opening the plugin to directly load data if a training hs juste been performed in the same
    # Svetlana's instance
    load_data_after_training()
    return prediction_widget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [Annotation, Training, Prediction]
