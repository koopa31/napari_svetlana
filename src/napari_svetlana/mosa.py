from skimage.io import imread, imsave
import os
import numpy as np
from PIL import ImageDraw, ImageFont, Image

line_nb = 8
col_nb = 8

path = "/home/clement/Bureau/res_cam_trous"

case = "im"

# im pour mosaique image ou cam pour l'autre
if case == "im":
    keyword = "rgb"
elif case == "cam":
    keyword = "visu"

onlyfiles = sorted([os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and
            keyword in f])[:64]

l = []
for i in range(line_nb):
    l.append([])
for i in range(line_nb):
    for j in range(col_nb):
           im = imread(onlyfiles[i + col_nb * j])

           if len(im.shape) != 3:
               im = np.stack((im,) * 3, axis=-1)
           im = Image.fromarray(im)
           im_editable = ImageDraw.Draw(im)
           text = os.path.splitext(os.path.split(onlyfiles[i + col_nb * j])[1])[0].split("_")[-1]
           font = ImageFont.truetype("arial.ttf", 8)
           im_editable.text((im.width // 9, 1), text, (255, 0, 0), font=font)
           im = np.asarray(im)
           l[i].append(im)

for i in range(col_nb):
    l[i] = np.hstack(l[i])

mosaic = np.vstack(l)

imsave(os.path.join(path, "mosaic_" + case + ".png"), mosaic)
