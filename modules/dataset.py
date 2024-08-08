from modules.utils import get_imagenet_transform
from modules.utils import get_files_recursive
from torchvision import transforms
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class SaliconDataset(Dataset):
    def __init__(self, input_path, folders=['train'], resize_ratio=1, base_size=[480, 640]):
        """
        This class matches between SALICON image files and its appropriate heatmap file and process them for training.
        The files are loaded into memory, then normalized and scaled so it could be used as a valid input to the model.
        Args:
            input_path: SALICON directory path
            folders: a list of sub directories to load. might contain 'train' and/or 'valid'
            resize_ratio: the downscaling ratio to use for all images/heatmaps. e.g: 0.5 will downscale the images by
                half
            base_size: the base size to use for all SALICON images (by default all the images are 480x640)
        """
        self.folders = folders
        self.resize_ratio = resize_ratio
        self.base_size = base_size
        self.input_path = Path(input_path)

        heatmap_transform = transforms.Lambda(lambda x: x / x.max())

        image_transform = get_imagenet_transform()

        image_normalizer = transforms.Compose([transforms.ToTensor(), image_transform])
        heatmap_normalizer = transforms.Compose([transforms.ToTensor(), heatmap_transform])

        if abs(1 - self.resize_ratio) > 0.0001:
            output_size = (np.array(self.base_size) * self.resize_ratio).astype(np.int)
        else:
            output_size = self.base_size

        self.image_normalizer = transforms.Compose([transforms.Resize(output_size), image_normalizer])
        self.heatmap_normalizer = transforms.Compose([transforms.Resize(output_size), heatmap_normalizer])

        images_arr = []
        heatmap_arr = []
        for folder in folders:
            curr_folder_images = get_files_recursive(str(self.input_path / 'images' / folder), 'jpg')

            for img_path in curr_folder_images:
                heatmap_name = Path(img_path).name.replace('jpg', 'png')
                heatmap_path = Path(self.input_path) / 'maps' / folder / heatmap_name

                images_arr.append(img_path)
                heatmap_arr.append(heatmap_path)

        self.pairs = list(zip(images_arr, heatmap_arr))

    def __getitem__(self, index):
        image_path, heatmap_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        heatmap = Image.open(heatmap_path).convert('L')

        image = self.image_normalizer(image)
        heatmap = self.heatmap_normalizer(heatmap)

        return (image, str(image_path)), (heatmap, str(heatmap_path))

    def __len__(self):
        return len(self.pairs)

