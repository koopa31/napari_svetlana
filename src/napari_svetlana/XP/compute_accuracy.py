from skimage.io import imread
import os
import pandas as pd
import numpy as np

"""
COMPUTATION OF ACCURACY OF CLASSIFICATION USING GROUNDTRUTH
"""

images_folder = "/home/cazorla/Images/Test papier svetlana/tube neural 3d/Test_depth"

groundtruth_path = "/home/cazorla/Images/Test papier svetlana/tube neural 3d/gt_new.tif"
mask_path = "/home/cazorla/Images/Test papier svetlana/tube neural 3d/Masks/" \
            "resampled_crop_nov 10th 2014 11ps dapi_Stitch-1_cp_masks.tif"

images_list = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith(".tif")])

gt = imread(groundtruth_path)
mask = imread(mask_path)

ROI_number = mask.max()

accuracy1_list = []
accuracy2_list = []
counter1_list = []
counter2_list = []
path_list = []

for i, image_path in enumerate(images_list):
    print("traitement image n°%i" % i)
    path_list.append(image_path)
    image = imread(image_path)

    # where image and gt are equal to 1
    class1 = (image == gt) & (gt == 1)
    mask1 = mask.copy()
    mask1 *= class1
    counter1 = len(np.unique(mask1)) - 1

    # where gt is equal to 1
    gt1 = (gt == 1)
    mask1 = mask.copy()
    mask1 *= gt1
    ROI1 = len(np.unique(mask1)) - 1

    # accuracy for label 1
    accuracy1 = counter1 / ROI1

    # where image and gt are equal to 1
    class2 = (image == gt) & (gt == 2)
    mask2 = mask.copy()
    mask2 *= class2
    counter2 = len(np.unique(mask2)) - 1

    # where gt is equal to 2
    gt2 = (gt == 2)
    mask2 = mask.copy()
    mask2 *= gt2
    ROI2 = len(np.unique(mask2)) - 1

    # accuracy for label 2
    accuracy2 = counter2 / ROI2

    accuracy1_list.append(accuracy1)
    accuracy2_list.append(accuracy2)

    counter1_list.append(ROI1 - counter1)
    counter2_list.append(ROI2 - counter2)


df = pd.DataFrame(list(zip(path_list, accuracy1_list, counter1_list, accuracy2_list, counter2_list)),
                  columns=['image name', 'accuracy1', "errors nb label1", 'accuracy2', "errors nb label2"])
df.to_excel(os.path.join(images_folder, "compute_accuracy.xlsx"), index=False)
