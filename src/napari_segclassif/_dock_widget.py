"""
SegClassif dock widget module
"""
import functools
import os
import pickle
import random
from typing import Any

import cv2
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
            res_dict = {"image_path": image_path, "labels_path": labels_path, "position_list": position_list,
                        "labels_list": labels_list}
            with open(path, "wb") as handle:
                pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

        global position_list
        position_list = []
        for i, prop in enumerate(mini_props):
            if prop.area != 0:
                if image.shape[2] <= 3:
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
                    position_list.append(prop.label)
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
                        position_list.append(prop.label)
        print(len(imagettes_list))

        global patch
        patch = (imagettes_list, maskettes_list, imagettes_contours_list)
        return patch

    @magicgui(
        #call_button='run segmentation',
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

    @magicgui(
        auto_call=True,
        layout='vertical',
        load_data_button=dict(widget_type='PushButton', text='Load data', tooltip='Load the image and the labels'),
        lr=dict(widget_type='LineEdit', label='Learning rata', value=0.01, tooltip='Learning rate'),
        nn=dict(widget_type='ComboBox', label='Network architecture', choices=networks_list, value="ResNet18",
                       tooltip='All the available network architectures'),
    )
    def training_widget(  # label_logo,
            viewer: Viewer,
            load_data_button,
            nn,
            lr,

    ) -> None:
        # Import when users activate plugin
        return

    @training_widget.load_data_button.changed.connect
    def _load_data(e: Any):
        path = QFileDialog.getOpenFileName(None, 'Open File', options=QFileDialog.DontUseNativeDialog)[0]
        with open(path, 'rb') as handle:
            b = pickle.load(handle)

        global image_path
        global labels_path
        global position_list
        global labels_list
        image_path = b["image_path"]
        labels_path = b["labels_path"]
        position_list = b["position_list"]
        labels_list = b["labels_list"]

        training_widget.viewer.value.add_image(imread(image_path))
        training_widget.viewer.value.add_labels(imread(labels_path))

        return

    return training_widget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return [Annotation, Training]
