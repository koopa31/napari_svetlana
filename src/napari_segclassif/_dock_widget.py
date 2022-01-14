"""
SegClassif dock widget module
"""
import random
from typing import Any

import cv2
from napari_plugin_engine import napari_hook_implementation

import time
import numpy as np

from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui

from skimage.measure import regionprops


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

counter = 0
labels_list = []
# logo = os.path.join(__file__, 'logo/logo_small.png')


def widget_wrapper():
    from napari.qt.threading import thread_worker

    @Viewer.bind_key('1')
    def print_names(viewer):
        global counter
        if counter < len(patch[2]) - 1:
            labels_list.append(1)
            viewer.layers.pop()
            counter += 1
            viewer.add_image(patch[2][counter])
            print("label 1", labels_list)
        elif counter == len(patch[2]) - 1:
            labels_list.append(1)
            viewer.layers.pop()
            counter += 1
            from skimage.io import imread
            viewer.add_image(imread("image_finish.png"))
            print("annotation over", labels_list)
        else:
            pass

    @Viewer.bind_key('2')
    def print_names(viewer):
        global counter
        if counter < len(patch[2]) - 1:
            labels_list.append(2)
            viewer.layers.pop()
            counter += 1
            viewer.add_image(patch[2][counter])
            print("label 2", labels_list)
        elif counter == len(patch[2]) - 1:
            labels_list.append(2)
            viewer.layers.pop()
            counter += 1
            from skimage.io import imread
            viewer.add_image(imread("image_finish.png"))
            print("annotation over", labels_list)
        else:
            pass

    @Viewer.bind_key('r')
    def print_names(viewer):
        global counter
        labels_list.pop()
        viewer.layers.pop()
        counter -= 1
        viewer.add_image(patch[2][counter])
        print("retour en arriere", labels_list)

    @thread_worker
    def generate_patches(viewer, imagettes_nb, patch_size):
        for im in viewer.choices:
            if "mask" in im.name:
                labels = im.data
            else:
                image = im.data

        half_patch_size = patch_size // 2
        contours_color = [0, 0, 255]

        props = regionprops(labels)
        random.shuffle(props)
        mini_props = props[:imagettes_nb]

        imagettes_list = []
        imagettes_contours_list = []
        maskettes_list = []

        for i, prop in enumerate(mini_props):
            if prop.area != 0:
                imagette = image[int(prop.centroid[0]) - half_patch_size:int(prop.centroid[0]) + half_patch_size,
                           int(prop.centroid[1]) - half_patch_size:
                           int(prop.centroid[1]) + half_patch_size]

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
                    imagette_contours[:, :, 1][contours != 0] = contours_color[2]

                imagettes_list.append(imagette)
                maskettes_list.append(maskette)
                imagettes_contours_list.append(imagette_contours)
        print(len(imagettes_list))

        global patch
        patch = (imagettes_list, maskettes_list, imagettes_contours_list)
        return patch

    @magicgui(
        #call_button='run segmentation',
        layout='vertical',
        patch_size=dict(widget_type='LineEdit', label='patch size', value=200, tooltip='extracted patch size'),
        patch_nb=dict(widget_type='LineEdit', label='patches number', value=10, tooltip='number of extracted patches'),
        labels_number=dict(widget_type='ComboBox', label='labels number', choices=labels_number, value=2,
                           tooltip='Number of possible labels'),
        extract_pacthes_button=dict(widget_type='PushButton', text='extract patches from image',
                                    tooltip='extraction of patches to be annotated from the segmentation mask'),
    )
    def annotation_widget(  # label_logo,
            viewer: Viewer,
            image_layer: Image,
            patch_size,
            patch_nb,
            extract_pacthes_button,
            labels_number,

    ) -> None:
        # Import when users activate plugin
        return

    def display_first_patch(patch):
        for i in range(0, len(annotation_widget.viewer.value.layers)):
            annotation_widget.viewer.value.layers.pop()
        annotation_widget.viewer.value.add_image(patch[2][0])

    @annotation_widget.extract_pacthes_button.changed.connect
    def _compute_diameter_shape(e: Any):
        patch_worker = generate_patches(annotation_widget.image_layer, int(annotation_widget.patch_nb.value), int(annotation_widget.patch_size.value))
        patch_worker.returned.connect(display_first_patch)
        patch_worker.start()
        print('patch extraction done')

    return annotation_widget


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'Annotation'}
