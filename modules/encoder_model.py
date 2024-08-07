import numpy as np
import torch
import torch.cuda
from torch import nn as nn
from modules.utils import get_imagenet_transform


class EncoderModule(nn.Module):
    def __init__(self, original_model, feature_indices):
        """
        Used to return different layer outputs from an existing model. This class can be used to extract the
        convolutional features of a network by plugging into it's last convolutional layer (e.g: VGG19)
        Args:
            original_model: nn.Module object.
            feature_indices: index of layers that should be returned as outputs from the encoder
        """
        super(EncoderModule, self).__init__()
        if torch.cuda.is_available():
            original_model = original_model.cuda()

        # saving it will make it trainable if necessary
        self.original_model = original_model

        self.feature_indices = feature_indices

    def forward(self, x):
        children_list = list(self.original_model.children())
        features_list = children_list[:min(self.feature_indices[-1] + 1, len(children_list))]

        outputs = []
        for feature_idx, feature in enumerate(features_list):
            x = feature(x)
            if feature_idx in self.feature_indices:
                outputs.append(x)
        return outputs

    def get_input_normalizer(self):
        if hasattr(self.original_model, 'get_input_normalizer'):
            return self.original_model.get_input_normalizer()
        else:
            return get_imagenet_transform()


def freeze_layer_groups(model, freeze_groups, train_bn=False):
    """
    Freeze layer groups, possibly excluding batchnorm layers
    Args:
        freeze_groups: list of booleans of length len(self.layer_groups()), indicating which groups to freeze
        train_bn: if True - always train batchnorm, even on freezed layers

    Returns:

    """
    bn_layers = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
    if freeze_groups is None:
        return
    layer_groups = model.layer_groups()
    assert len(freeze_groups) == len(layer_groups)
    frozen_layers = []
    for layers, freeze in zip(layer_groups, freeze_groups):
        for l in layers:
            l_name = None
            if isinstance(l, str):
                # layer_groups returning dict format (with names)
                l_name = l
                l = layers[l_name]
            train_layer = not freeze
            if freeze:
                frozen_layers.append((l_name, l))
            if isinstance(l, nn.Parameter):
                l.requires_grad = train_layer
                continue
            l.train(train_layer)
            if train_bn and isinstance(l, bn_layers):
                train_layer = True
            for p_name, p in l.named_parameters():
                p.requires_grad = train_layer
    return frozen_layers
