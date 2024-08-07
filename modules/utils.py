import os.path
from torch.utils.data import SubsetRandomSampler
import numpy as np
from torch.utils.data import ConcatDataset
from torchvision import transforms
import scipy.misc
import skimage.transform
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def resize_center_bias(center_bias, new_shape):
    min_val = center_bias.min()
    max_val = center_bias.max()
    center_bias_normalized = (center_bias - min_val) / (max_val - min_val)
    center_bias_resized = skimage.transform.resize(center_bias_normalized, new_shape, order=0, mode='edge')
    center_bias_resized = center_bias_resized * (max_val - min_val) + min_val
    center_bias_resized -= scipy.misc.logsumexp(center_bias_resized)
    return center_bias_resized


def get_imagenet_transform():
    return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def get_reverse_imagenet_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    new_mean = [-m/s for m,s in zip(mean,std)]
    new_std = [1/s for s in std]
    return transforms.Normalize(mean=new_mean, std=new_std)


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def get_files_recursive(file_dir, ext=None):
    """
    runs get_files on all subdirectories recursively and returns a concatenated file list
    Args:
        file_dir: path to a directory
        ext: desired extension (e.g: '.jpg'). If none, returns any file
    Returns:
        list of full file paths
    """
    if not os.path.exists(file_dir):
        return []
    files = get_files(file_dir, ext)
    entries = os.listdir(file_dir)
    for entry in entries:
        entry_path = os.path.join(file_dir, entry)
        if os.path.isdir(entry_path):
            files += get_files_recursive(entry_path, ext)
    return files


def get_files(file_dir, ext=None):
    """
    get all file paths (full paths) in a directory with specific extension
    Args:
        file_dir: path to a directory
        ext: desired extension (e.g: '.jpg'). If none, returns any file
    Returns:
        list of full file paths
    """
    if not os.path.exists(file_dir):
        return []
    files = os.listdir(file_dir)
    paths = []
    for x in files:
        if ext is None:
            paths.append(os.path.join(file_dir, x))
        if isinstance(ext, str) and x.lower().endswith(ext):
            paths.append(os.path.join(file_dir, x))
        if isinstance(ext, list):
            for curr_ext in ext:
                if x.lower().endswith(curr_ext):
                    paths.append(os.path.join(file_dir, x))

    return paths


def get_subset_sampler(dataset, max_images, shuffle=False):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle:
        np.random.shuffle(indices)
    indices = indices[:min(max_images, len(indices))]
    sampler = SubsetRandomSampler(indices)
    return sampler


def embed_image(img, heatmap, heatmap_weight=0.7, norm_heatmap=False):
    height, width, _ = img.shape
    if norm_heatmap:
        hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        hm_norm = 1 - hm_norm
    else:
        hm_norm = heatmap
    hm_norm = cv2.applyColorMap(cv2.resize((hm_norm * 255).astype(np.uint8), (width, height)), cv2.COLORMAP_JET)
    result = hm_norm * heatmap_weight + img * (1 - heatmap_weight)
    return result


def plot_img(img, mask=None, mask_weight=0.7, title="", norm_heatmap=False, figsize=None, norm=True):


    if isinstance(img, Image.Image):
        img = np.array(img)

    if len(img.shape) < 2:
        raise Exception("image shape is invalid: should be 2d or 3d image")

    img = np.squeeze(img)

    if len(img.shape) > 2:
        if img.shape[2] < 3:
            raise Exception("image shape is invalid, color channels should be of the size of 3")

    if img.dtype == bool:
        img = img.astype(int)
    if img.dtype == np.float16:
        img = img.astype(float)

    if mask is not None:
        img = embed_image(img, mask, mask_weight, norm_heatmap)

    img_norm = img.astype(np.uint8)
    if norm:
        if img.max() > img.min():
            img_norm = (img - img.min()) / (img.max() - img.min())
        else:
            img_norm = np.ones(img.shape) * img.max()

    interpolation = None
    dpi = None
    if figsize == 'auto':  # select it so that the image is displayed with 1:1 pixel mapping
        dpi = 80
        ypixels, xpixels = img_norm.shape[:2]
        interpolation = 'none'

        # Make a figure big enough to accomodate an axis of xpixels by ypixels
        # as well as the ticklabels, etc...
        figsize = xpixels / dpi, ypixels / dpi

        fig = plt.figure(figsize=figsize, dpi=dpi)

        # Make the axis the right size...
        ax = fig.add_axes([0, 0, 1, 1])
        # ax.imshow(np.random.random((xpixels, ypixels)), interpolation='none')
    else:
        fig = plt.figure(figsize=figsize, dpi=dpi)

    plt.title(title)
    plt.imshow(img_norm, interpolation=interpolation)
    plt.show(block=True)
