import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from modules.utils import get_imagenet_transform, resize_center_bias, num_flat_features
from modules.encoder_model import EncoderModule


import torch.nn as nn
from enum import Enum

class VGG_LEVEL(Enum):
    VGG_11 = 0
    VGG_19 = 1

class OUTPUT_MODE(Enum):
    SOFTMAX = 0
    SIGMOID = 1


# use_gpu = torch.cuda.is_available()

class DeepGazeVGG(nn.Module):
    def __init__(self, output_upsample_ratio=None,
                 initial_gaussian_std=0.0217, vgg_level=VGG_LEVEL.VGG_19, output_mode=OUTPUT_MODE.SOFTMAX,
                 freeze_encoder=True):
        """
        Model based on Deep Gaze 2 paper ("Understanding Low- and High-Level Contributions to Fixation Prediction")
        The implementation changes try to follow Twitter's paper (https://arxiv.org/pdf/1801.05787.pdf, "Faster gaze
        prediction with dense networks and Fisher pruning").
        Basically it involves extracting VGG19/VGG11 features, adding a "readout network" on top of them (a set of
        1x1 convolutional filters) and output a probability map for gaze prediction.

        Additional input to the model is a center bias - a log probability heat map that describes the probability of a
        gaze to be upon a pixel in the dataset. this center bias is actually the mean heatmap. notice that center bias
        is relative to the dataset.

        The readout network output is blurred by a gaussian (the std of the gaussian is a learned parameter) and
        then the center bias is added to the output. the result is passed through a softmax/sigmoid activation layer in
        order to create a heatmap (notice that the softmax normalization is over the entire image, which means that
        the sum of the result should be 1)

        Args:
            output_upsample_ratio: the scale-up that should be done to the output of the readout network. If None, no
                upsampling is done (e.g: output_upsample_ratio=8 will upsample the features by 8)
            initial_gaussian_std: the initialization of the std of the blurring kernel
            vgg_level: can determine if using vgg11 or vgg19 features. see VGG_LEVEL enum.
            output_mode: Either softmax or sigmoid. see OUTPUT_MODE enum.
            freeze_encoder: if True, the encoder (i.e: VGG) is frozen and not trainable
        """
        super(DeepGazeVGG, self).__init__()

        self.output_upsample_ratio = output_upsample_ratio
        self.output_mode = OUTPUT_MODE(output_mode)
        self.vgg_level = VGG_LEVEL(vgg_level)

        if self.vgg_level == VGG_LEVEL.VGG_11:
            print("using vgg11 encoder")
            self.vgg_encoder = EncoderModule(models.vgg11(pretrained=True).features, [18])
            num_readout_input_features = 512
        else:
            print("using vgg19 encoder")
            self.vgg_encoder = EncoderModule(models.vgg19(pretrained=True).features, [28, 29, 31, 32, 35])
            num_readout_input_features = 2560  # 512*5

        if freeze_encoder:
            for param in self.vgg_encoder.parameters():
                param.requires_grad = False

        self.inst_norm = nn.InstanceNorm2d(num_readout_input_features)
        self.readout_conv1 = nn.Conv2d(num_readout_input_features, 16, (1, 1), bias=False)  # todo:bias is false?
        self.readout_conv2 = nn.Conv2d(16, 32, (1, 1), bias=False)
        self.readout_conv3 = nn.Conv2d(32, 2, (1, 1), bias=False)
        self.readout_conv4 = nn.Conv2d(2, 1, (1, 1), bias=False)
        self.PReLU1 = nn.PReLU()
        self.PReLU2 = nn.PReLU()
        self.PReLU3 = nn.PReLU()
        self.PReLU4 = nn.PReLU()

        self.soft_max = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.sigma = nn.Parameter(torch.from_numpy(np.array([initial_gaussian_std])))
        self.window_radius = 40

        self.center_bias = None

        if output_upsample_ratio is not None:
            self.output_resize = nn.Upsample(scale_factor=output_upsample_ratio, mode="bilinear")
        else:
            self.output_resize = None

    def process_center_bias(self, dataset_center_bias, input_resize_ratio=None):
        """
        resizing the center bias to comply with the input size.
        Args:
            dataset_center_bias: np.array, the center bias of the dataset
            input_resize_ratio: the desired resize ratio of the dataset image

        Returns:
            np.array - center bias for the resized dataset
        """
        scale = 1.0
        # should scale the center bias according to the desired input size
        if input_resize_ratio is not None:
            scale = input_resize_ratio

        # should renormalize the center_bias to be a valid log probability
        if scale < 1:
            scale_resized = [dataset_center_bias.shape[0] * scale, dataset_center_bias.shape[1] * scale]
            dataset_center_bias_resized = (resize_center_bias(dataset_center_bias, scale_resized)).astype(
                dataset_center_bias.dtype)
        else:
            dataset_center_bias_resized = dataset_center_bias

        bias_tensor = torch.from_numpy(dataset_center_bias_resized)
        if torch.cuda.is_available():
            bias_tensor = bias_tensor.cuda()
        return bias_tensor

    def set_center_bias(self, dataset_center_bias, input_resize_ratio=None):
        """
        Sets the center bias for the dataset
        Args:
            dataset_center_bias: np.array, center bias matrix, see __init__ remarks about the desired size of the matrix
            input_resize_ratio: the resize ratio that is going to be used before feeding the input images to the network
                this parameter is crucial for proper use of the center bias
        """
        print("set_center_bias: input_resize_ratio = ", input_resize_ratio)
        bias_tensor = self.process_center_bias(dataset_center_bias, input_resize_ratio)
        self.center_bias = dict()
        if torch.cuda.is_available():
            bias_tensor = bias_tensor.cuda()
        self.center_bias = Variable(bias_tensor, requires_grad=False)

    def forward(self, x):
        output_features = self.vgg_encoder(x)

        x = torch.cat(output_features, 1)
        x = self.inst_norm(x)

        x = self.PReLU1(self.readout_conv1(x))
        x = self.PReLU2(self.readout_conv2(x))
        x = self.PReLU3(self.readout_conv3(x))
        x = self.PReLU4(self.readout_conv4(x))

        if self.output_resize is not None:
            x = self.output_resize(x)

        x = gaussian_blur(x, self.sigma, self.window_radius)
        if self.center_bias is not None:
            if self.output_mode == OUTPUT_MODE.SOFTMAX:
                # (center_bias_rows - x_rows)/2 is the symmetric buffer size between x and the horizontal edges of
                # center_bias. The same goes for the columns
                center_row_start = int((self.center_bias.shape[0] - x.shape[2]) / 2)
                center_row_end = center_row_start + x.shape[2]
                center_col_start = int((self.center_bias.shape[1] - x.shape[3]) / 2)
                center_col_end = center_col_start + x.shape[3]

                x = x + self.center_bias[center_row_start:center_row_end, center_col_start:center_col_end]

        if self.output_mode == OUTPUT_MODE.SOFTMAX:
            orig_shape = x.size()
            x = x.view(-1, num_flat_features(x))
            x = self.soft_max(x)
            x = x.view(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3])
        elif self.output_mode == OUTPUT_MODE.SIGMOID:
            x = self.sigmoid(x)

        return x


def gaussian_blur(input, sigma, window_radius=40):
    """
    Filter input with a Gaussian using mode `nearest`.
    input is expected to be three-dimensional of type n times x times y

    Args:
        input: Tensor, some feature tensor that should undergo blurring (should be of size [batch size, c,h,w])
        sigma: Variable, the standard deviation of the gaussian kernel (Since it is a variable, you can train this parameter)
        window_radius: the window radius that should be used for the gaussian kernel.

    Returns:
        Blurred Tensor.
    """
    # Construction of 1d kernel
    # Work around some strange theano bug

    #input_size = input.size()
    #sigma_pix = sigma * min(input_size[2], input_size[3])

    filter_1d_tensor = torch.arange(0, 2 * window_radius + 1) - window_radius
    filter_1d = Variable(filter_1d_tensor.type(sigma.data.type()), requires_grad=False)

    filter_1d = torch.exp(-0.5 * filter_1d ** 2 / sigma ** 2)
    filter_1d = filter_1d / filter_1d.sum()
    filter_1d = filter_1d.type_as(input)

    # channel first - (batch_size, 3, h, w)

    # from shape [2*window_radius-1] to [1,2*window_radius-1]
    W = filter_1d.unsqueeze(1)

    # from shape [2*window_radius-1] to [2*window_radius-1, 1]
    W2 = filter_1d.unsqueeze(0)

    # from c,h,w -> 1,c,h,w
    # blur_input = input.unsqueeze(0)

    # from shape [k,1] to [1,1,k,1] (e.g: batch_size, channel, 1, k)
    filter_W = W.unsqueeze(0).unsqueeze(0)
    # from shape [1,k] to [1,1,1,k] (e.g: batch_size, channel, k, 1)
    filter_W2 = W2.unsqueeze(0).unsqueeze(0)

    # Construction of filter pipeline
    blur_input_start = input[:, :, :1, :]
    blur_input_end = input[:, :, -1:, :]

    padded_input = torch.cat([blur_input_start] * window_radius + [input] + [blur_input_end] * window_radius,
                             dim=2)

    blur_op = F.conv2d(padded_input, filter_W)

    cropped_output1 = blur_op
    cropped_output1_start = blur_op[:, :, :, :1]
    cropped_output1_end = blur_op[:, :, :, -1:]

    padded_cropped_input = torch.cat(
        [cropped_output1_start] * window_radius + [cropped_output1] + [cropped_output1_end] * window_radius, dim=3)

    cropped_output2 = F.conv2d(padded_cropped_input, filter_W2)

    return cropped_output2

