import matplotlib.cm as mpl_color_map
import cv2
import os
import torch
import torch.nn as nn
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def weight2color(weight_im):
    """
    colorize weight map

    Args:
        weight_im ([(0,1) array of BxHxW]): weight/mask maps
    """
    color_map = mpl_color_map.get_cmap('rainbow')
    heatmap = color_map(weight_im) * 255
    return heatmap


def tensor2rgb(tensor):
    tensor = tensor.numpy().squeeze().transpose(0, 2, 3, 1)

    # Scale between 0-255 to visualize

    tensor = np.uint8((tensor+1.)/2 * 255)
    # tensor = np.uint8(Image.fromarray(tensor)
    #                     .resize((256, 256), Image.ANTIALIAS))
    return tensor


def heatmaped_img(tensor_im, weight_im, to_tensor=True):
    img = tensor2rgb(tensor_im)
    output = []
    for i in range(img.shape[0]):
        w_im = weight_im[i]
        im = img[i]
        heatmap = weight2color(w_im)
        att = cv2.cvtColor(heatmap.astype(np.uint8), cv2.COLOR_RGBA2RGB)
        att = cv2.resize(att, (im.shape[0], img.shape[1]))
        dst = cv2.addWeighted(im.astype(np.float32), 0.6,
                              att.astype(np.float32), 0.4, 0)
        output.append(np.clip(dst.astype(np.uint8), 0, 255))

    output = np.array(output)

    if to_tensor:
        output = torch.from_numpy(output)
        output = output.permute(0, 3, 1, 2)
        output = output/255*2 - 1
        # print(output.size())

    return output


def plot_fc_weight(sample, weight):
    if sample.size(1) == 1:
        sample = sample.repeat(1, 3, 1, 1)  # gray 2 rgb

    plot_list = []
    plot_list.append(sample)

    B, C, H, W = sample.size()

    weight -= torch.min(weight)
    weight /= torch.max(weight)
    weight = weight.contiguous().view(H, W).unsqueeze(0)
    weight = weight.repeat(B, 1, 1).numpy()

    plot_list.append(heatmaped_img(sample, weight))
    plot_list = torch.cat(plot_list, dim=0)

    return plot_list
