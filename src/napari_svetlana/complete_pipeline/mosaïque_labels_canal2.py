import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw, ImageFont, Image
from skimage.measure import regionprops

"""
Création de la mosaïque des labels du canal DAPI en couleurs, plus enregistrement des masques en couleur.
"""


def generate_mosa_edu(folder):
    images_folder = os.path.join(folder, "Images")
    pred_folder = os.path.join(folder, "Predictions_noyaux_edu")
    labels_folder = os.path.join(folder, "Color_labels")

    # Création du dossier de labels
    if os.path.isdir(labels_folder) is False:
        os.mkdir(labels_folder)

    images_path = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if
                           os.path.isfile(os.path.join(images_folder, f)) and os.path.join(images_folder, f).endswith(".tif")
                          and "d1" in os.path.join(images_folder, f)])
    pred_path = sorted([os.path.join(pred_folder, f) for f in os.listdir(pred_folder) if
                           os.path.isfile(os.path.join(pred_folder, f)) and os.path.join(pred_folder, f).endswith(".tif")])


    fields_list = []
    pred_fields_list = []
    # On crée une liste de fichiers par puit (9 champs)
    for i in range(42):
        fields_list.append(images_path[0 + i * 9: 9 + i * 9])
        pred_fields_list.append(pred_path[0 + i * 9: 9 + i * 9])

    wells_mosaics = []

    for i in range(len(fields_list)):
        lines_list = []
        # Tous les champs d'un puit
        images_list = [cv2.imread(file) for file in fields_list[i]]
        masks_list = [np.array(Image.open(file)) for file in pred_fields_list[i]]

        # Egalisation et padding des images du puit pour avoir  une séparation entre les images
        for j in range(0, len(images_list)):
            images_list[j] = ((images_list[j] - images_list[j].min()) / (
                        images_list[j].max() - images_list[j].min()) * 255).astype("uint8")
            #Création de l'overlay de couleur
            image = images_list[j].astype("uint16").copy()
            labels = masks_list[j].astype("uint16").copy()

            output = image.copy()
            overlay = np.zeros((labels.shape[0], labels.shape[1], 3)).astype("uint16")
            overlay = image.copy()
            props = regionprops(labels)

            overlay[labels == 1] = [0, 255, 0]
            overlay[labels == 2] = [255, 0, 0]

            """for lab in props:
                if labels[int(lab.centroid[0]), int(lab.centroid[1])] == 1:
                    color = [0, 255, 0]
                else:
                    color = [255, 0, 0]
                for coord in lab.coords:
                    overlay[coord[0], coord[1], 0] = color[0]
                    overlay[coord[0], coord[1], 1] = color[1]
                    overlay[coord[0], coord[1], 2] = color[2]"""
            alpha = 0.4
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

            images_list[j] = output.astype("uint8")

            cv2.imwrite(os.path.join(labels_folder, os.path.splitext(os.path.split(fields_list[i][j])[1])[0] + "_labels.TIF"),
                        cv2.cvtColor(images_list[j], cv2.COLOR_BGR2RGB))

            # Padding poiur ajout de contours dans la mosaique
            images_list[j] = np.pad(images_list[j], ((10, 10), (10, 10), (0, 0)), constant_values=255)

            # Ecriture de la date et du nom de l'image sur l'image
            images_list[j] = Image.fromarray(images_list[j])
            image_editable = ImageDraw.Draw(images_list[j])
            image_name = os.path.split(fields_list[i][j])[1]
            well_name = image_name.split("_")[-1][:3].capitalize()
            field_name = image_name.split("_")[-1][3:6].capitalize()

            font = ImageFont.truetype("arial.ttf", 40)
            image_editable.text((images_list[j].width // 2, 15), field_name, (255, 0, 0), font=font)
            image_editable.text((images_list[j].width // 2 - int(0.25 * images_list[j].width // 2) -
                                 image_editable.textsize(well_name)[0], 15), well_name, (255, 0, 0), font=font)

            images_list[j] = np.asarray(images_list[j]).astype("uint8")


        # Création de la mosaique des champs d'un puit
        for k in range(3):
            lines_list.append(np.hstack(images_list[0 + 3 * k: 3 + 3 * k]))
        wells_mosaics.append(np.vstack(lines_list))

    lines_stack = []

    for i in range(6):
        lines_stack.append(np.hstack(wells_mosaics[0 + 7 * i: 7 + 7 * i]))

    final_mosaic = np.vstack(lines_stack)

    final_mosaic = Image.fromarray(final_mosaic)

    mosaic_folder = os.path.split(images_folder)[0]
    mosaic_name = os.path.join(mosaic_folder, mosaic_folder.split("/")[-1] + "_mosaique_edu.tif")

    final_mosaic.save(mosaic_name, quality=25)
    print('mosaique calculée')
