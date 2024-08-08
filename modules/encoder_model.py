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
        """
        Runs the underlying model (self.original_model) and picks the output of a specific layers
        """
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

