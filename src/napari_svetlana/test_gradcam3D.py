import matplotlib.pyplot as plt
import torch
from Grad_Cam.grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from skimage.io import imread
from torchvision import transforms
from CNN2D import CNN2D
from skimage.measure import regionprops
import numpy as np
from Prediction3DDataset import Prediction3DDataset
import json
from torch.utils.data import DataLoader


#model1 = resnet50(pretrained=True)

b = torch.load("/home/cazorla/Images/Test_fluidite3D/Svetlana/training_500.pth")

model = b["model"]

target_layers = [model.cnn_layers]
rgb_img = imread("/home/cazorla/Images/Test_fluidite3D/Images/neural_tube_3D.tif")
if len(rgb_img.shape) == 2:
    rgb_img = np.stack((rgb_img,) * 3, axis=-1)
mask = imread("/home/cazorla/Images/Test_fluidite3D/Masks/neural_tube_3D_cp_masks.tif")

patch_size = 30

pad_image = np.pad(rgb_img, ((patch_size // 2 + 1, patch_size // 2 + 1),
                   (patch_size // 2 + 1, patch_size // 2 + 1),
                   (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
pad_labels = np.pad(mask, ((patch_size // 2 + 1, patch_size // 2 + 1),
                    (patch_size // 2 + 1, patch_size // 2 + 1),
                    (patch_size // 2 + 1, patch_size // 2 + 1)), mode="constant")
props = regionprops(mask)

with open("/home/cazorla/Documents/Codes/napari_svetlana/src/napari_svetlana/Config.json", 'r') as f:
    config_dict = json.load(f)

data = Prediction3DDataset(pad_image, pad_labels, props, patch_size // 2, b["norm_type"], "cuda", config_dict)

batch_size = 1
prediction_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
for i, local_batch in enumerate(prediction_loader):
    if i == 900:
        input_tensor = local_batch
        #input_tensor[1, :, :, :, :] = input_tensor[0, :, :, :, :]
        """rgb_img = imread("/home/cazorla/Téléchargements/dog_cat(1).jfif")
        input_tensor = transforms.ToTensor()(rgb_img).to("cuda")
        # Create an input tensor image for your model..
        # Note: input_tensor can be a batch tensor with several images!
    
        model = resnet50(pretrained=True)
        target_layers = [model.layer4]"""

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

        # You can also use it within a with statement, to make sure it is freed,
        # In case you need to re-create it inside an outer loop:
        # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
        #   ...

        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category
        # will be used for every image in the batch.
        # Here we use ClassifierOutputTarget, but you can define your own custom targets
        # That are, for example, combinations of categories, or specific outputs in a non standard model.

        # Pour savoir qel label a été créé
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

        np_arr = np.zeros((input_tensor.shape[2], input_tensor.shape[3], input_tensor.shape[4]))

        np_arr = input_tensor[0, 0, :, :, :].cpu().detach().numpy().copy()

        from skimage.io import imsave
        c = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min()) * 255
        imsave("/home/cazorla/Bureau/" + str(i) + "3dcam.tif", c)
        imsave("/home/cazorla/Bureau/" + str(i) + "patch3D.tif", np_arr)
        """
        np_arr = (np_arr - np_arr.min()) / (np_arr.max() - np_arr.min())
        grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())

        visualization = show_cam_on_image(np_arr.astype(np.float32), grayscale_cam, use_rgb=True)

        from skimage.io import imsave

        imsave("/home/clement/Bureau/visu" + str(i) + "_index_" + str(index[0].item()) + ".jpeg", visualization)
        imsave("/home/clement/Bureau/rgb" + str(i) + "_index_" + str(index[0].item()) + ".jpeg", input_tensor[0, 0, :, :, ].cpu().detach().numpy())


        plt.figure(1)
        plt.imshow(visualization)
        plt.show()"""
