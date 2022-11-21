import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw, ImageFont, Image

"""
Script pour la génération de mosaïques pour le contole qualité de la classification de cellules mortes et vivantes
"""


def generate_mosa_raw_images(folder):
    images_folder = os.path.join(folder, "Images")

    images_list = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if
                          os.path.isfile(os.path.join(images_folder, f)) and os.path.join(images_folder, f).endswith(
                              ".tif") and "d0.tif" in os.path.join(images_folder, f)])
    fields_list = []
    pred_fields_list = []
    # On crée une liste de fichiers par puit (9 champs)
    for i in range(42):
        fields_list.append(images_list[0 + i * 9: 9 + i * 9])

    wells_mosaics = []

    for i in range(len(fields_list)):
        lines_list = []
        # Tous les champs d'un puit
        images_list = [cv2.imread(file) for file in fields_list[i]]
        # Egalisation et padding des images du puit pour avoir  une séparation entre les images
        for j in range(0, len(images_list)):
            images_list[j] = ((images_list[j] - images_list[j].min()) / (
                        images_list[j].max() - images_list[j].min()) * 255).astype("uint8")
            images_list[j] = np.pad(images_list[j], ((3, 3), (3, 3), (0, 0)), constant_values=255)
            images_list[j] = cv2.resize(images_list[j], (images_list[j].shape[1] // 4, images_list[j].shape[0] // 4),
                                        interpolation=cv2.INTER_CUBIC)

            # Ecriture de la date et du nom de l'image sur l'image
            images_list[j] = Image.fromarray(images_list[j])
            image_editable = ImageDraw.Draw(images_list[j])
            image_name = os.path.split(fields_list[i][j])[1]
            well_name = image_name.split("_")[-1][:3].capitalize()
            field_name = image_name.split("_")[-1][3:6].capitalize()

            font = ImageFont.truetype("arial.ttf", 10)
            image_editable.text((images_list[j].width // 2, 15), field_name, (255, 0, 0), font=font)
            image_editable.text((images_list[j].width // 2 - int(0.25 * images_list[j].width // 2) -
                                 image_editable.textsize(well_name)[0], 15), well_name, (255, 0, 0), font=font)

            images_list[j] = np.asarray(images_list[j]).astype("uint8")


        # Création de la mosaique des champs d'un puit
        for k in range(3):
            lines_list.append(np.hstack(images_list[0 + 3 * k: 3 + 3 * k]))
        wells_mosaics.append(np.vstack(lines_list))
        wells_mosaics[-1] = np.pad(wells_mosaics[-1], ((3, 3), (3, 3), (0, 0)), constant_values=255)

    lines_stack = []

    for i in range(6):
        lines_stack.append(np.hstack(wells_mosaics[0 + 7 * i: 7 + 7 * i]))
        #cv2.imwrite("/home/clement/Bureau/line" + str(i) + ".tif", lines_stack[i])

    final_mosaic = np.vstack(lines_stack)

    final_mosaic = Image.fromarray(final_mosaic)

    mosaic_folder = os.path.split(images_folder)[0]
    mosaic_name = os.path.join(mosaic_folder, mosaic_folder.split("/")[-1] + "_mosaique.tif")

    final_mosaic.save(mosaic_name, quality=25)
    print('mosaique calculée')
