from cv2 import imread, imwrite, resize
import os

"""path = "Z:\Projets\I3DOPPORTUNITES\I3DOP049UNIVERCELL\Acquisitions\\2022-08-HCD-manip7-Harmine\\2022_08_12\sphero-redim-EdU"
res_path = os.path.join(path, "resultats")
if os.path.isdir(res_path) is False:
    os.mkdir(res_path)

onlyfiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and
             os.path.join(path, f).endswith(".tif")]


for p in onlyfiles:
    image = imread(p)
    resized_image = resize(image, dsize=(image.shape[0] * 2, image.shape[1] * 2))
    imwrite(os.path.join(res_path, os.path.split(p)[1]), resized_image)"""

path = "D:\\imactiv3d\\Documents\\Users\\Pascale\\HCD-EdU_resultat\\2022_08_12\\PlaqueA\\crops"
res_path = os.path.join(path, "resultats")
if os.path.isdir(res_path) is False:
    os.mkdir(res_path)
onlyfiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and
             os.path.join(path, f).endswith(".tif")]

for p in onlyfiles:
    image = imread(p)
    if image.shape[0] == 1752:
        image = image[136:1752-136, 136:1752-136, :]
        imwrite(os.path.join(res_path, os.path.split(p)[1]), image)