import torch
from torch import nn
import torch.nn.functional as F
from gen import sine
import numpy as np


def pixel_shuffle(input, upscale_x_factor, upscale_y_factor):
    r"""Rearranges elements in a tensor of shape :math:`[*, C*r^2, H, W]` to a
    tensor of shape :math:`[C, H*r, W*r]`.

    See :class:`~torch.nn.PixelShuffle` for details.

    Args:
        input (Tensor): Input
        upscale_factor (int): factor to increase spatial resolution by

    Examples::

        >>> ps = nn.PixelShuffle(3)
        >>> input = torch.empty(1, 9, 4, 4)
        >>> output = ps(input)
        >>> print(output.size())
        torch.Size([1, 1, 12, 12])
    """
    batch_size, channels, in_height, in_width = input.size()
    channels //= upscale_x_factor * upscale_y_factor

    out_height = in_height * upscale_x_factor
    out_width = in_width * upscale_y_factor

    input_view = input.contiguous().view(
        batch_size, channels, upscale_x_factor, upscale_y_factor,
        in_height, in_width)

    shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width)


def get_sine_mask(period, width, length, phase, batch):
    a = sine(period, length, phase)
    a = a[np.newaxis, :].repeat(width, axis=0)
    a = a[np.newaxis, :].repeat(batch, axis=0)

    return torch.Tensor(a)


# TODO: pixel unshuffle?!


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 15, kernel_size=(1, 9), padding=(0, 4))
        self.bn1 = nn.BatchNorm2d(15)

        self.conv2 = nn.Conv2d(5, 15, kernel_size=(1, 9), padding=(0, 4))
        self.bn2 = nn.BatchNorm2d(15)

        self.conv3 = nn.Conv2d(15, 15, kernel_size=(1, 9), padding=(0, 4))
        self.bn3 = nn.BatchNorm2d(15)

        self.conv4 = nn.Conv2d(15, 15, kernel_size=1)
        self.bn4 = nn.BatchNorm2d(15)

        self.c1x1_l = nn.Conv2d(15, 3, kernel_size=1)
        self.c1x1_p = nn.Conv2d(15, 2, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.max_pool2d(x, (1, 2))
        x = x.view(10, 5, 3, -1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        label = self.c1x1_l(x)

        x = F.max_pool2d(x, (1, 2))

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = F.max_pool2d(x, (1, 2))

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        phase = self.c1x1_p(F.max_pool2d(x, (1, 2)))

        return label, phase


class Decoder(nn.Module):
    def __init__(self, batch_size=10):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(3, 20, kernel_size=(1, 9), padding=(0, 4))
        self.bn1 = nn.BatchNorm2d(20)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(16, 20, kernel_size=(1, 9), padding=(0, 4))
        self.bn2 = nn.BatchNorm2d(20)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(10, 20, kernel_size=(1, 9), padding=(0, 4))
        self.c1x1 = nn.Conv2d(10, 1, kernel_size=1)

        self.c1x1_p = nn.Conv2d(2, 6, kernel_size=1)

        self.masks = []

        self.masks.append(get_sine_mask(10, 3, 512, 0., batch_size))
        self.masks.append(get_sine_mask(20, 3, 512, 0., batch_size))
        self.masks.append(get_sine_mask(30, 3, 512, 0., batch_size))
        self.masks.append(get_sine_mask(10, 3, 512, 0.5 * np.pi, batch_size))
        self.masks.append(get_sine_mask(20, 3, 512, 0.5 * np.pi, batch_size))
        self.masks.append(get_sine_mask(30, 3, 512, 0.5 * np.pi, batch_size))

    def forward(self, label, phase):
        x = self.conv1(label)
        x = self.bn1(x)
        x = self.relu1(x)
        x = pixel_shuffle(x, 1, 2)

        h = self.c1x1_p(phase.repeat(1, 1, 1, 16))
        # print(h.shape, self.masks[0].shape)
        h[:, 0, :, :] *= self.masks[0]
        h[:, 1, :, :] *= self.masks[1]
        h[:, 2, :, :] *= self.masks[2]
        h[:, 3, :, :] *= self.masks[3]
        h[:, 4, :, :] *= self.masks[4]
        h[:, 5, :, :] *= self.masks[5]

        # print(x.shape, h.shape)
        x = torch.cat((x,F.avg_pool2d(h, (1, 8))), dim=1)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = pixel_shuffle(x, 1, 2)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = pixel_shuffle(x, 1, 2)

        return self.c1x1(x)


if __name__ == '__main__':
    a = get_sine_mask(10, 3, 296, 0., 10)
