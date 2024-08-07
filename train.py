import os
import argparse
import warnings
import json
import numpy as np
import torch.cuda
from modules.utils import get_subset_sampler
from modules.model import DeepGazeVGG
from modules.dataset import SaliconDataset
from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim
import torch.nn as nn
import pathlib


def train_model(args, config):
    ## Model configuration
    model = DeepGazeVGG(**config['model'])
    center_bias = None
    if args.center_bias_path is not None:
        center_bias = np.load(args.center_bias_path).astype(np.float32)

    ## Dataset configuration
    train_dataset = SaliconDataset(args.data_path, ['train'])
    valid_dataset = SaliconDataset(args.data_path, ['valid'])

    if config.get('max_images',None) is not None:
        train_sampler = get_subset_sampler(train_dataset, config['max_images'])
        valid_sampler = get_subset_sampler(valid_dataset, config['max_images'])
    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = RandomSampler(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=6)
    val_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], sampler=valid_sampler, num_workers=6)

    if center_bias is not None:
        print("using center bias!!")
        model.set_center_bias(center_bias, train_dataset.resize_ratio)

    if torch.cuda.is_available():
        model = model.cuda()

    model = model.train()
    optimizer = optim.SGD(model.parameters(),
                          lr=config['lr'],
                          momentum=config['momentum'],
                          weight_decay=config['weight_decay'],
                          nesterov=True)
    criterion = nn.BCELoss()

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    train_proc(model=model,
               device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
               dataloaders={'train': train_loader, 'val': val_loader},
               criterion=criterion, optimizer=optimizer,
               num_epochs=config['num_epochs'],
               output_path=pathlib.Path(args.output_path))


def train_proc(model, device, dataloaders, criterion, optimizer, num_epochs, output_path, check_point=1):
    model.train()

    for epoch in range(num_epochs):

        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 20)

        for phase in ['train', 'val']:
            running_loss = 0
            if phase == 'train':
                model.train()
            else:
                model.eval()

            cnt = 0
            for img, heatmap in dataloaders[phase]:
                img = img[0].to(device)
                heatmap = heatmap[0].to(device)

                with torch.autograd.set_grad_enabled(phase == 'train'):
                    outputs = model(img)
                    loss = criterion(outputs, heatmap)
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                cnt = cnt + 1
                if cnt == len(dataloaders[phase]):
                    print("\r {} Complete: {:.2f}".format(phase, cnt / len(dataloaders[phase])), end='\n')
                else:
                    print("\r {} Complete: {:.2f}".format(phase, cnt / len(dataloaders[phase])), end="")

                running_loss += loss.item() * img.size(0)

            epoch_loss = running_loss / len(dataloaders[phase])

            print("{} Loss: {}".format(phase, epoch_loss))

        if num_epochs % check_point == 0:
            torch.save(model.state_dict(), output_path / 'salicon_model.pth')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='path to configuration json')
    parser.add_argument('--output_path', type=str, required=True, help='training result path (model, checkpoints etc.)')
    parser.add_argument('--data_path', type=str, required=True, help='SALICON images folder')
    parser.add_argument('--max_images', type=str, required=False, help='maximum number of images to use', default=None)
    parser.add_argument('--center_bias_path', type=str, required=False, help='optional center bias numpy file')

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    else:
        warnings.warn('output_path already exist!')

    if not os.path.exists(args.data_path):
        raise Exception("data_path doesn't exist")

    with open(args.config_file) as data_file:
        config = json.load(data_file)

    if args.max_images is not None:
        config['max_images'] = args.max_images

    train_model(args, config)

    print("Done!")


if __name__ == '__main__':
    main()
