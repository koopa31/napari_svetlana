"""
Svetlana dock widget module
"""
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


labels_number = [('2', 2), ('3', 3), ('4', 4), ('5', 5), ('6', 6), ('7', 7), ('8', 8), ('9', 9)]
networks_list = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "AlexNet", "DenseNet121",
                 "DenseNet161", "DenseNet169", "DenseNet201", "CustomCNN2D"]
losses_list = ["CrossEntropy", "L1Smooth", "BCE", "Distance", "L1", "MSE"]

counter = 0
# counter of images to be annotated
image_counter = 0
total_counter = 0

global_labels_list = []
global_im_path_list = []
global_lab_path_list = []
global_mini_props_list = []

# list of all regionprops to be exported in a specific binary file
props_to_be_saved = []


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
            global counter, total_counter
            print("key is ", key)
            if (int(annotation_widget.labels_nb.value) < key) is False:
                if counter < len(props) - 1:
                    global_labels_list[image_counter].append(key)

                    global_mini_props_list[image_counter].append(
                        {"centroid": props[indexes[counter]].centroid, "coords": props[indexes[counter]].coords,
                         "label": props[indexes[counter]].label})
                    if case == "2D" or case == "multi2D":
                        progression_mask[
                            props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1]] = key
                        counter += 1
                        total_counter += 1

                        # focus on the next object to annotate
                        if double_click is False:
                            viewer.camera.zoom = zoom_factor
                            viewer.camera.center = (0, int(props[indexes[counter]].centroid[0]),
                                                    int(props[indexes[counter]].centroid[1]))
                            viewer.camera.zoom = zoom_factor + 10 ** -8
                        # deletion of the old contours and drawing of the new one
                        circle_mask[circle_mask != 0] = 0
                        circle_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1]] = 1
                        eroded_contours = cv2.erode(np.uint16(circle_mask), np.ones((5, 5), np.uint8))
                        eroded_labels = circle_mask - eroded_contours

                        # Pyramidal representation of the contours to enhance the display speed

                        annotation_widget.viewer.value.layers[-1].data_raw[0][
                            annotation_widget.viewer.value.layers[-1].data_raw[0] != 0] = 0
                        annotation_widget.viewer.value.layers[-1].data_raw[0][eroded_labels == 1] = 1

                        for i in range(1, len(annotation_widget.viewer.value.layers[-1].data_raw)):
                            annotation_widget.viewer.value.layers[-1].data_raw[i] = \
                                cv2.resize(annotation_widget.viewer.value.layers[-1].data_raw[0], (
                                    annotation_widget.viewer.value.layers[-1].data_raw[0].shape[0] // 2 ** i,
                                    annotation_widget.viewer.value.layers[-1].data_raw[0].shape[1] // 2 ** i))

                    else:
                        progression_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1],
                                         props[indexes[counter]].coords[:, 2]] = key
                        counter += 1
                        total_counter += 1

                        # focus on the next object to annotate
                        if double_click is False:
                            viewer.camera.zoom = zoom_factor
                            viewer.camera.center = (int(props[indexes[counter]].centroid[0]),
                                                    int(props[indexes[counter]].centroid[1]),
                                                    int(props[indexes[counter]].centroid[2]))
                            annotation_widget.viewer.value.dims.current_step = (int(props[indexes[counter]].centroid[0]),
                                                                                annotation_widget.viewer.value.dims.current_step[
                                                                                    1],
                                                                                annotation_widget.viewer.value.dims.current_step[
                                                                                    2])
                            viewer.camera.zoom = zoom_factor + 10 ** -8
                        # deletion of the old contours and drawing of the new one
                        circle_mask[circle_mask != 0] = 0
                        circle_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1],
                                    props[indexes[counter]].coords[:, 2]] = 1
                        annotation_widget.viewer.value.layers[-1].data = circle_mask

                    annotation_widget.viewer.value.layers.selection.active = annotation_widget.viewer.value.layers[
                        image_layer_name]

                    print("label 1", global_labels_list[image_counter])
                    viewer.status = str(counter) + " images processed over " + str(len(props)) + " (" + \
                                    str(total_counter) + " over the whole batch)"
                elif counter == len(props) - 1:
                    global_labels_list[image_counter].append(key)
                    global_mini_props_list[image_counter].append(
                        {"centroid": props[indexes[counter]].centroid, "coords": props[indexes[counter]].coords,
                         "label": props[indexes[counter]].label})
                    progression_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1]] = key
                    counter += 1
                    total_counter += 1
                    from skimage.io import imread
                    viewer.layers.clear()
                    viewer.add_image(imread("https://bitbucket.org/koopa31/napari_package_images/raw/"
                                            "a9fda1dd3361880162474cf0b30119b1e188f53c/image_finish.png"))
                    print("annotation over", global_labels_list[image_counter])
                    viewer.status = str(counter) + " images processed over " + str(len(props)) + " (" + \
                                    str(total_counter) + " over the whole batch)"
                    show_info("Annotation over, press 1 to save the result")
                else:
                    # Saving of the annotation result in a binary file

                    path = QFileDialog.getSaveFileName(None, 'Save File', options=QFileDialog.DontUseNativeDialog)[0]
                    res_dict = {"image_path": global_im_path_list, "labels_path": global_lab_path_list,
                                "regionprops": global_mini_props_list, "labels_list": global_labels_list,
                                "patch_size": annotation_widget.patch_size.value}
                    torch.save(res_dict, path)

        return set_label

    @Viewer.bind_key('r')
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
            if double_click is False:
                viewer.camera.zoom = zoom_factor
                viewer.camera.center = (0, int(props[indexes[counter]].centroid[0]),
                                        int(props[indexes[counter]].centroid[1]))
                viewer.camera.zoom = zoom_factor + 10 ** -8
            circle_mask[circle_mask != 0] = 0
            circle_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1]] = 1
            eroded_contours = cv2.erode(np.uint16(circle_mask), np.ones((5, 5), np.uint8))
            eroded_labels = circle_mask - eroded_contours

            # Pyramidal representation of the contours to enhance the display speed
            annotation_widget.viewer.value.layers[-1].data_raw[0][
                annotation_widget.viewer.value.layers[-1].data_raw[0] != 0] = 0
            annotation_widget.viewer.value.layers[-1].data_raw[0][eroded_labels == 1] = 1

            for i in range(1, len(annotation_widget.viewer.value.layers[-1].data_raw)):
                annotation_widget.viewer.value.layers[-1].data_raw[i] = \
                    cv2.resize(annotation_widget.viewer.value.layers[-1].data_raw[0], (
                        annotation_widget.viewer.value.layers[-1].data_raw[0].shape[0] // 2 ** i,
                        annotation_widget.viewer.value.layers[-1].data_raw[0].shape[1] // 2 ** i))

        else:
            progression_mask[props[indexes[counter]].coords[:, 0], props[indexes[counter]].coords[:, 1],
                             props[indexes[counter]].coords[:, 2]] = 0
            if double_click is False:
                viewer.camera.zoom = zoom_factor
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
            annotation_widget.viewer.value.layers[-1].data = circle_mask

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
            from .CustomDialog import CustomDialog
            diag = CustomDialog()
            diag.exec()
            case = diag.get_case()
            print(case)

        global props, props_to_be_saved
        props = regionprops(labels)

        # The regionprops are shuffled to be randomly labeled
        global indexes
        indexes = np.random.permutation(np.arange(0, len(props))).tolist()

        present = []
        for prop in props_to_be_saved:
            if prop["props"] == props:
                present.append(True)
            else:
                present.append(False)

        if True not in present:
            props_to_be_saved.append({"props": props, "indexes": indexes})

        return props, zoom_factor

    @magicgui(
        auto_call=True,
        layout='vertical',
        load_images_button=dict(widget_type='PushButton', text='LOAD IMAGES',
                                tooltip='Load images'),
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
        save_button=dict(widget_type='PushButton', text='Save annotation', tooltip='Save annotation'),

        save_regionprops_button=dict(widget_type='PushButton', text='Save objects statistics', tooltip='Save the '
                                                                                                       'properties of the annotated objects in a binary file, loadable using torch.load'),
        generate_im_labs_button=dict(widget_type='PushButton', text='Save masks of labels', tooltip='Save one '
                                                                                                    'per attributed label'),
        show_labs=dict(widget_type='CheckBox', text='Show labeled objects', tooltip='Show labeled objects'),
        click_annotate=dict(widget_type='CheckBox', text='Click to annotate', tooltip='Click to annotate'),
    )
    def annotation_widget(  # label_logo,
            viewer: Viewer,
            load_images_button,
            previous_button,
            image_index_button,
            next_button,
            estimate_size_button,
            patch_size,
            extract_pacthes_button,
            labels_nb,
            save_button,
            save_regionprops_button,
            generate_im_labs_button,
            show_labs,
            click_annotate

    ) -> None:
        # Create a black image just so layer variable exists
        if len(viewer.layers) == 0:
            viewer.add_image(np.zeros((1000, 1000)))
        # Import when users activate plugin
        global layer, double_click
        # By default, we do not annotate clicking
        double_click = False
        for l in viewer.layers:
            if "mask" not in l.name:
                layer = l

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
            if double_click is True:
                if case == "2D":
                    ind = labels[int(event.position[0]), int(event.position[1])] - 1
                elif case == "multi2D":
                    ind = labels[int(event.position[1]), int(event.position[2])] - 1
                elif case == "3D":
                    ind = labels[int(event.position[0]), int(event.position[1]), int(event.position[2])] - 1
                else:
                    ind = labels[int(event.position[1]), int(event.position[2]), int(event.position[3])] - 1

                try:
                    indexes.remove(ind)
                    indexes.insert(counter, ind)
                    print('position', event.position)
                    show_info("Choose a label for that object")
                except ValueError:
                    show_info("please click on a valid object")

    @annotation_widget.load_images_button.changed.connect
    def load_images(e: Any):
        """
        Loads the folder containing the images to annotates and displays the first one
        @param e:
        @return:
        """
        # Gets the folder url and the two subfolder containing the images and the masks
        global images_folder, masks_folder, parent_path

        parent_path = QFileDialog.getExistingDirectory(None, 'Open Folder', options=QFileDialog.DontUseNativeDialog)

        images_folder = os.path.join(parent_path, "Images")
        masks_folder = os.path.join(parent_path, "Masks")

        # Gets the list of images and masks
        global image_path_list, mask_path_list, global_im_path_list, global_lab_path_list, global_labels_list, \
            global_mini_props_list, mini_props_list

        image_path_list = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder)])
        mask_path_list = sorted([os.path.join(masks_folder, f) for f in os.listdir(masks_folder)])
        global_im_path_list = image_path_list.copy()
        global_lab_path_list = mask_path_list.copy()

        for i in range(0, len(image_path_list)):
            global_labels_list.append([])
            global_mini_props_list.append([])

        # Deletion of remaining image and displaying of the first uimage of the list
        annotation_widget.viewer.value.layers.clear()
        annotation_widget.viewer.value.add_image(imread(image_path_list[image_counter]))
        annotation_widget.viewer.value.add_labels(imread(mask_path_list[image_counter]))
        annotation_widget.viewer.value.layers[1].name = "mask"

        # original zoom factor to correct when annotating
        global old_zoom
        old_zoom = annotation_widget.viewer.value.camera.zoom

    @annotation_widget.image_index_button.changed.connect
    def set_image_index(e: Any):

        global counter, image_counter

        if int(e) > len(global_im_path_list) - 1:
            show_info("Too high index")
            annotation_widget.image_index_button.value = len(global_im_path_list)
        elif int(e) < 1:
            show_info("Too low index")
            annotation_widget.image_index_button.value = 1

        image_counter = int(annotation_widget.image_index_button.value) - 1

        counter = len(global_labels_list[image_counter])

        annotation_widget.viewer.value.layers.clear()
        annotation_widget.viewer.value.add_image(imread(os.path.join(images_folder, image_path_list[image_counter])))
        annotation_widget.viewer.value.add_labels(imread(os.path.join(masks_folder, mask_path_list[image_counter])))
        annotation_widget.viewer.value.layers[1].name = "mask"

    @annotation_widget.next_button.changed.connect
    def next_image(e: Any):
        """
        Loads the next image to annotate
        @param e:
        @return:
        """
        global image_counter, counter, labels_list, mini_props_list

        if image_counter < len(global_im_path_list) - 1:

            # Reinitialization of counter for next image
            counter = len(global_labels_list[image_counter + 1])

            image_counter += 1
            # Update of the image index
            annotation_widget.image_index_button.value = image_counter + 1

            annotation_widget.viewer.value.layers.clear()
            annotation_widget.viewer.value.add_image(imread(os.path.join(images_folder, image_path_list[image_counter])))
            annotation_widget.viewer.value.add_labels(imread(os.path.join(masks_folder, mask_path_list[image_counter])))
            annotation_widget.viewer.value.layers[1].name = "mask"
        else:
            show_info("No more images")

    @annotation_widget.previous_button.changed.connect
    def previous_image(e: Any):
        """
        Loads the next image to annotate
        @param e:
        @return:
        """
        global image_counter, counter, labels_list, mini_props_list
        if image_counter > 0:
            image_counter -= 1
            # Update of the image index
            annotation_widget.image_index_button.value = image_counter + 1

            annotation_widget.viewer.value.layers.clear()
            annotation_widget.viewer.value.add_image(imread(os.path.join(images_folder, image_path_list[image_counter])))
            annotation_widget.viewer.value.add_labels(imread(os.path.join(masks_folder, mask_path_list[image_counter])))
            annotation_widget.viewer.value.layers[1].name = "mask"

            # Reinitialization of counter for next image
            counter = len(global_labels_list[image_counter])

        else:
            show_info("No previous image")

    @annotation_widget.click_annotate.changed.connect
    def click_to_annotate(e: Any):
        """
        Activates click to annotate option
        @param e: boolean value of the checkbox
        @return:
        """
        global double_click
        if e is True:
            double_click = True
            # select the image so the user can click on it
            annotation_widget.viewer.value.layers.selection.active = annotation_widget.viewer.value.layers[
                image_layer_name]
        else:
            double_click = False
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
                    for ind, prop in enumerate(global_mini_props_list[image_counter]):
                        progression_mask[prop["coords"][:, 0], prop["coords"][:, 1]] = \
                        global_labels_list[image_counter][
                            ind]
                else:
                    progression_mask = np.zeros_like(circle_mask)
            else:
                image_layer_name = layer.name

        # Contour of object to annotate
        if case == "2D" or case == "multi2D":
            circle_mask[props[current_index].coords[:, 0], props[current_index].coords[:, 1]] = 1
            eroded_contours = cv2.erode(np.uint16(circle_mask), np.ones((5, 5), np.uint8))
            eroded_labels = circle_mask - eroded_contours
            # Pyramidal representation of the contours to enhance the display speed
            pyramid = [eroded_labels]
            for i in range(1, 6):
                pyramid.append(cv2.resize(eroded_labels, (eroded_labels.shape[0] // 2 ** i,
                                                          eroded_labels.shape[1] // 2 ** i)))
            annotation_widget.viewer.value.add_labels(pyramid)
        else:
            circle_mask[props[current_index].coords[:, 0], props[current_index].coords[:, 1],
                        props[current_index].coords[:, 2]] = 1
            annotation_widget.viewer.value.add_labels(circle_mask)
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
                pyramid.append(cv2.resize(progression_mask, (progression_mask.shape[0] // 2 ** i,
                                                             progression_mask.shape[1] // 2 ** i)))

            if e is True:
                annotation_widget.viewer.value.add_labels(pyramid, name="progression_mask")
            else:
                annotation_widget.viewer.value.layers.pop()
        else:
            if e is True:
                annotation_widget.viewer.value.add_labels(progression_mask, name="progression_mask")
            else:
                annotation_widget.viewer.value.layers.pop()

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
        patch_worker = generate_patches(annotation_widget.viewer.value.layers, int(annotation_widget.patch_size.value))
        patch_worker.returned.connect(display_first_patch)
        patch_worker.start()
        print('patch extraction done')

        # Disabling the estimate size and patch size in the gui so it is not change while annotating
        annotation_widget.estimate_size_button.enabled = False
        annotation_widget.patch_size.enabled = False
        # We make the labels mask invisible so it does not bother not annotate
        annotation_widget.viewer.value.layers[1].visible = False

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

            if props[:counter + 1][0].coords.shape[1] == 3:
                for i in range(0, counter):
                    props_list[j].append({"position": props[indexes[i]].label, "coords": props[indexes[i]].coords,
                                          "centroid": props[indexes[i]].centroid, "area": props[indexes[i]].area,
                                          "label": int(global_labels_list[j][i])})
            else:
                for i in range(0, counter):
                    props_list[j].append({"position": props[indexes[i]].label, "coords": props[indexes[i]].coords,
                                          "centroid": props[indexes[i]].centroid,
                                          "eccentricity": props[indexes[i]].eccentricity, "area": props[indexes[i]].area,
                                          "perimeter": props[indexes[i]].perimeter, "label": int(global_labels_list[j][i])})
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
                                                 props[indexes[i]].coords[:, 2]] = props[indexes[i]].label
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

    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def get_image_patch(image_list, mask_list, region_props_list, labels_list, torch_type, case):
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
                    imagette_mask[imagette_mask == region_props[i]["label"]] = max_type_val

                    concat_image = np.zeros((imagette.shape[0], imagette.shape[1], image.shape[2] + 1))
                    concat_image[:, :, :-1] = imagette
                    concat_image[:, :, -1] = imagette_mask
                    # Normalization of the image
                    concat_image = (concat_image - concat_image.min()) / (concat_image.max() - concat_image.min())

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

    def train(viewer, image, mask, region_props_list, labels_list, nn_type, loss_func, lr, epochs_nb, rot, h_flip,
              v_flip, prob, batch_size, saving_ep, training_name, model=None):
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
                   "DenseNet161": "densenet161", "DenseNet169": "densenet169", "DenseNet201": "densenet201",
                   "CustomCNN2D": "CNN2D"}
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
                        model.features[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                      bias=False)

                else:
                    model = CNN2D(max(labels_list), 4)

            elif len(image.shape) == 4:
                case = "multi3D"
                image = np.transpose(image, (1, 2, 3, 0))
                mask = np.transpose(mask, (1, 2, 0))
                model = CNN3D(max(labels_list), image.shape[3] + 1)

            else:
                from .CustomDialog import CustomDialog
                diag = CustomDialog()
                diag.exec()
                case = diag.get_case()

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
                        model = CNN2D(max(labels_list), 4)

                elif case == "3D":
                    model = CNN3D(max(labels_list), 2)

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
                                     labels_list, torch_type, case)
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
        """
        Function triggered by the load data button which aims at loading the needed information from the annotation,
        in order to train the NN
        @param e:
        @return:
        """
        training_widget.viewer.value.layers.clear()
        path = QFileDialog.getOpenFileName(None, 'Open File', options=QFileDialog.DontUseNativeDialog)[0]

        b = torch.load(path)

        global image_path_list
        global labels_path_list
        global region_props_list
        global labels_list
        global patch_size
        global image
        global mask
        image_path_list = b["image_path"]
        labels_path_list = b["labels_path"]
        region_props_list = b["regionprops"]
        labels_list = b["labels_list"]
        patch_size = int(b["patch_size"])

        image = imread(image_path_list[0])
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        mask = imread(labels_path_list[0])
        training_widget.viewer.value.add_image(image)
        training_widget.viewer.value.add_labels(mask)

        return

    @training_widget.load_custom_model_button.changed.connect
    def load_custom_model():
        """
        Function triggered by load custom model button to load a pretrained model of the user
        @return:
        """
        path = QFileDialog.getOpenFileName(None, 'Open File', options=QFileDialog.DontUseNativeDialog)[0]
        checkpoint = torch.load(path)
        global model
        model = checkpoint["model"]
        show_info("Model loaded successfully")

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
                 training_widget.loss.value, float(training_widget.lr.value),
                 int(training_widget.epochs.value), training_widget.rotations.value,
                 training_widget.h_flip.value, training_widget.v_flip.value,
                 float(training_widget.prob.value), int(training_widget.b_size.value),
                 int(training_widget.saving_ep.value), str(training_widget.training_name.value), model)
        else:

            training_worker = thread_worker(train, progress={"total": int(training_widget.epochs.value)})(
                training_widget.viewer, image, mask, region_props_list, labels_list,
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
            if double_click is True:
                imagette_contours[mask == lab] = int(key)
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

        imagette_contours[prop.coords[:, 0], prop.coords[:, 1]] = list_pred[i].item()
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

        imagette_contours[prop.coords[:, 0], prop.coords[:, 1], prop.coords[:, 2]] = list_pred[i].item()
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

            data = PredictionDataset(pad_image, pad_labels, props, patch_size // 2, max)

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

        # Save the result automatically
        res_name = "prediction_" + os.path.split(image_path_list[int(prediction_widget.image_index_button.value)])[1]
        imsave(os.path.join(res_folder, res_name), imagette_contours.astype(np.uint8))

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

        for ind in range(0, len(image_path_list)):
            image = imread(image_path_list[ind])
            labels = imread(mask_path_list[ind])
            props = regionprops(labels)

            import time
            start = time.time()
            compteur = 0
            global imagette_contours

            try:
                max = np.iinfo(image.dtype).max
            except:
                max = np.finfo(image.dtype).max

            if len(image.shape) == 2:
                image = np.stack((image,) * 3, axis=-1)

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

                data = PredictionDataset(pad_image, pad_labels, props, patch_size // 2, max)

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

            stop = time.time()
            print("temps de traitement", stop - start)
            show_info(str(np.sum(compteur)) + " objects remaining over " + str(len(props)))
            print(str(np.sum(compteur)) + " objects remaining over " + str(len(props)))

            # Save the result automatically
            res_name = "prediction_" + os.path.split(image_path_list[ind])[1]
            imsave(os.path.join(res_folder, res_name), imagette_contours.astype(np.uint8))

            show_info("prediction of image " + os.path.split(image_path_list[ind])[1] + " done")

    @magicgui(
        auto_call=True,
        layout='vertical',
        batch_size=dict(widget_type='LineEdit', label='Batch size', value=100, tooltip='Batch size'),
        load_network_button=dict(widget_type='PushButton', text='Load network', tooltip='Load weights of the NN'),
        load_data_button=dict(widget_type='PushButton', text='Load data', tooltip='Load images to process'),
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
        bound=dict(widget_type='CheckBox', text='Show boundaries only', tooltip='Show boundaries only'),
        generate_im_labs_button=dict(widget_type='PushButton', text='Save masks of labels', tooltip='Save one '
                                                                                                    'per attributed label'),
        save_regionprops_button=dict(widget_type='PushButton', text='Save objects statistics', tooltip='Save the '
                                                                                                       'properties of the annotated objects in a binary file, loadable using torch.load'),
        click_annotate=dict(widget_type='CheckBox', text='Click to change label', tooltip='Click to change label'),
    )
    def prediction_widget(  # label_logo,
            viewer: Viewer,
            load_network_button,
            load_data_button,
            previous_button,
            image_index_button,
            next_button,
            batch_size,
            launch_prediction_button,
            launch_batch_prediction_button,
            bound,
            save_regionprops_button,
            generate_im_labs_button,
            click_annotate

    ) -> None:
        # Import when users activate plugin
        # We generate the functions to add a label when a key i pressed
        for i in range(0, 10):
            viewer.bind_key(str(i), on_pressed(i), overwrite=True)

        if len(viewer.layers) > 0:
            global layer, double_click
            # By default, we do not annotate clicking
            double_click = True

            layer = [x for x in prediction_widget.viewer.value.layers if x.name == "image"][0]

            @layer.mouse_double_click_callbacks.append
            def label_clicking(layer, event):
                """
                When click to annotate option is activated, retrieves the coordinate of the clicked object to give him a
                label
                @param layer:
                @param event: Qt click event
                @return:
                """
                global lab
                if double_click is True:
                    if case == "2D":
                        lab = mask[int(event.position[0]), int(event.position[1])]
                    elif case == "multi2D":
                        lab = mask[int(event.position[1]), int(event.position[2])]
                    elif case == "3D":
                        lab = mask[int(event.position[0]), int(event.position[1]), int(event.position[2])]
                    else:
                        lab = mask[int(event.position[1]), int(event.position[2]), int(event.position[3])]
                    show_info("Choose a label for that object")

    @prediction_widget.click_annotate.changed.connect
    def click_to_annotate(e: Any):
        """
        Activates click to annotate option
        @param e: boolean value of the checkbox
        @return:
        """
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
        path = QFileDialog.getOpenFileName(None, 'Open File', options=QFileDialog.DontUseNativeDialog)[0]
        """
        with open(path, 'rb') as handle:
            b = pickle.load(handle)
        """
        b = torch.load(path)

        global model
        global patch_size

        model = b["model"].to("cuda")
        patch_size = b["patch_size"]
        model.eval()
        show_info("NN loaded successfully")

    @prediction_widget.load_data_button.changed.connect
    def _load_data(e: Any):
        """
        Function triggered by the load data button to load the weights of the neural network
        @param e:
        @return:
        """

        path = QFileDialog.getExistingDirectory(None, 'Open Folder', options=QFileDialog.DontUseNativeDialog)

        # Result folder
        global res_folder
        res_folder = os.path.join(path, "Predictions")
        if os.path.isdir(res_folder) is False:
            os.mkdir(res_folder)

        global images_folder, masks_folder
        images_folder = os.path.join(path, "Images")
        masks_folder = os.path.join(path, "Masks")

        global image_path_list, mask_path_list
        image_path_list = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder)])
        mask_path_list = sorted([os.path.join(masks_folder, f) for f in os.listdir(masks_folder)])

        global image, mask

        image = imread(image_path_list[0])
        if len(image.shape) == 2:
            image = np.stack((image,) * 3, axis=-1)
        mask = imread(mask_path_list[0])
        prediction_widget.viewer.value.add_image(image)
        prediction_widget.viewer.value.add_labels(mask)

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

        return

    @prediction_widget.image_index_button.changed.connect
    def set_image_index(e: Any):
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
        prediction_widget.viewer.value.add_image(image)
        prediction_widget.viewer.value.add_labels(mask)

    @prediction_widget.previous_button.changed.connect
    def load_previous_image(e: Any):
        if int(prediction_widget.image_index_button.value) > 1:
            prediction_widget.image_index_button.value = int(prediction_widget.image_index_button.value) - 1
        else:
            show_info("No previous image")

    @prediction_widget.next_button.changed.connect
    def load_next_image(e: Any):
        if int(prediction_widget.image_index_button.value) < len(image_path_list):
            prediction_widget.image_index_button.value = int(prediction_widget.image_index_button.value) + 1
        else:
            show_info("No more images")

    def display_result(image):
        """
        Displays the classified mask once the neural network prediction is over
        @param image: classified mask to display as an overlay
        @return:
        """
        prediction_widget.viewer.value.add_labels(image)
        # Chose of the colours
        prediction_widget.viewer.value.layers[1].name = "Classified labels"
        if len(np.unique(prediction_widget.viewer.value.layers[1].data)) == 3:
            prediction_widget.viewer.value.layers[1].color = {1: "green", 2: "red"}

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
        @return:
        """

        if "edge_im" not in globals():
            # computation of the cells segmentation edges
            eroded_contours = cv2.erode(np.uint16(mask), np.ones((7, 7), np.uint8))
            eroded_labels = mask - eroded_contours

            # Removing the inside of the cells in the binary result using the edges mask computed just before
            global edge_im
            edge_im = imagette_contours.copy().astype(np.uint8)
            edge_im[eroded_labels == 0] = 0
        if e is True:
            pyramidal_edge_im = [edge_im]
            for i in range(1, 6):
                pyramidal_edge_im.append(cv2.resize(edge_im, (edge_im.shape[0] // 2 ** i,
                                                              edge_im.shape[1] // 2 ** i)))
            prediction_widget.viewer.value.layers.pop()
            prediction_widget.viewer.value.add_labels(pyramidal_edge_im)
            if len(np.unique(prediction_widget.viewer.value.layers[1].data)) == 3:
                prediction_widget.viewer.value.layers[1].color = {1: "green", 2: "red"}
        else:
            pyramidal_imagette_contours = [imagette_contours.astype(np.uint8)]
            for i in range(1, 6):
                pyramidal_imagette_contours.append(
                    cv2.resize(imagette_contours.astype(np.uint8), (imagette_contours.shape[0] // 2 ** i,
                                                                    imagette_contours.shape[
                                                                        1] // 2 ** i)))
            prediction_widget.viewer.value.layers.pop()
            prediction_widget.viewer.value.add_labels(pyramidal_imagette_contours)
            if len(np.unique(prediction_widget.viewer.value.layers[1].data)) == 3:
                prediction_widget.viewer.value.layers[1].color = {1: "green", 2: "red"}

    @prediction_widget.save_regionprops_button.changed.connect
    def save_regionprops():
        """
        Saves the properties of the labelled connected components in a binary file using Pytorch
        @return:
        """

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
        """
        Saves a mask for each class
        @return:
        """
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
