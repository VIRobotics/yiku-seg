"""This module has some useful functions"""

import os
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
def save_images(images, image_paths, data):
    images = Image.fromarray(images)
    label = data["label"][0]
    file_name = data["path"][0].split("/")[-1]
    if not os.path.exists(image_paths):
        os.mkdir(image_paths)
    image_paths = os.path.join(image_paths, label + file_name)
    images.save(image_paths)


def convert2img(image, imtype=np.uint8):
    if not isinstance(image, np.ndarray):
        if isinstance(image, torch.Tensor):
            image = image.data
        else:
            return image
        image = image.cpu().numpy()
        assert len(image.squeeze().shape) < 4
    if image.dtype != np.uint8:
        image = (np.transpose(image.squeeze(), (1, 2, 0)) * 0.5 + 0.5) * 255
    return image.astype(imtype)


def plt_show(img):
    img = torchvision.utils.make_grid(img.cpu().detach())
    img = img.numpy()
    if img.dtype != "uint8":
        img_numpy = img * 0.5 + 0.5
    plt.figure(figsize=(10, 20))
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))
    plt.show()


def compare_images(real_img, generated_img, threshold=0.4):
    generated_img = generated_img.type_as(real_img)
    diff_img = np.abs(generated_img - real_img)
    real_img = convert2img(real_img)
    generated_img = convert2img(generated_img)
    diff_img = convert2img(diff_img)

    threshold = (threshold*0.5+0.5)*255
    diff_img[diff_img <= threshold] = 0

    anomaly_img = np.zeros_like(real_img)
    anomaly_img[:, :, :] = real_img
    anomaly_img[np.where(diff_img>0)[0], np.where(diff_img>0)[1]] = [200, 0, 0]
    #anomaly_img[:, :, 0] = anomaly_img[:, :, 0] + 10.0 * np.mean(diff_img, axis=2)

    return convert2img(anomaly_img)