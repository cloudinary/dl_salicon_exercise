import os
import argparse
import warnings
import json
import numpy as np
import torch.cuda
from modules.utils import get_subset_sampler, plot_img, get_reverse_imagenet_transform
from modules.model import DeepGazeVGG
from modules.dataset import SaliconDataset
from torch.utils.data import DataLoader, RandomSampler
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def load_model_from_file(weight_file, config):
    model = DeepGazeVGG(**config['model'])
    state_dict = torch.load(weight_file)
    model.load_state_dict(state_dict)
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def evaluate_model(args, config, should_plot=True):
    model = load_model_from_file(args.weight_file, config)
    test_dataset = SaliconDataset(args.data_path, ['valid'])

    if config['max_images'] is not None:
        test_sampler = get_subset_sampler(test_dataset, config['max_images'])
    else:
        test_sampler = RandomSampler(test_dataset)

    train_loader = DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=0)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    trans = get_reverse_imagenet_transform()
    for img, heatmap in tqdm(train_loader):
        print("img[1] = ", img[1][0])
        img_tensor = img[0].to(device)
        print("img_tensor.shape = ", img_tensor.shape)
        with torch.no_grad():
            output = model(img_tensor)

        pred_heatmap = output.detach().cpu().numpy()[0, 0, :, :]

        img_tensor = trans(img_tensor.detach().cpu())
        img_np = img_tensor.numpy()[0].transpose([1, 2, 0])
        cropped_img = crop(img_np, pred_heatmap, aspect_ratio=args.aspect_ratio)

        if should_plot:
            plot_img(img_np, title="original")
            plot_img(img_np * 255, mask=pred_heatmap, title="heatmap")
            plot_img(cropped_img*255, title="crop")

        Image.fromarray((img_np*255).astype(np.uint8)).save(Path(args.output_path) / Path(img[1][0]).name)
        Image.fromarray((pred_heatmap * 255).astype(np.uint8)).save(Path(args.output_path) / Path(heatmap[1][0]).name)
        Image.fromarray((cropped_img * 255).astype(np.uint8)).save(Path(args.output_path) / ("crop_"+Path(img[1][0]).name))


def crop(img, heatmap, aspect_ratio=1):
    h, w = img.shape[:2]

    # ar is w/h

    if h > w:
        if aspect_ratio < 1:
            # crop height>crop width
            crop_w = w
            crop_h = int(w / aspect_ratio)
        else:
            # crop height<=crop width
            crop_h = h
            crop_w = int(h * aspect_ratio)
    else:
        # h<=w
        if aspect_ratio < 1:
            # crop height>crop width
            crop_h = h
            crop_w = int(h * aspect_ratio)
        else:
            # crop height<=crop width
            crop_w = w
            crop_h = int(w / aspect_ratio)

    max_row, max_col, max_score = -1, -1, -1
    for row in range(0, img.shape[0] - crop_h + 1):
        row_end = row + crop_h
        for col in range(0, img.shape[1] - crop_w + 1):
            col_end = col + crop_w
            curr_score = heatmap[row:row_end, col:col_end].sum()
            if curr_score > max_score:
                max_row = row
                max_col = col
                max_score = curr_score

    return img[max_row:max_row + crop_h, max_col:max_col + crop_w]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='path to configuration json')
    parser.add_argument('--weight_file', type=str, required=True, help='trained pth file')
    parser.add_argument('--output_path', type=str, required=True, help='results path')
    parser.add_argument('--data_path', type=str, required=True, help='SALICON images folder')
    parser.add_argument('--max_images', type=str, required=False, help='maximum number of images to use', default=3)
    parser.add_argument('--center_bias_path', type=str, required=False, help='optional center bias numpy file')
    parser.add_argument('--aspect_ratio', type=str, required=False, help='aspect ratio for crop', default=0.5)

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    else:
        warnings.warn('output_path already exist!')

    with open(args.config_file) as data_file:
        config = json.load(data_file)

    if args.max_images is not None:
        config['max_images'] = args.max_images

    evaluate_model(args, config, should_plot=False)

    print("Done!")


if __name__ == '__main__':
    main()
