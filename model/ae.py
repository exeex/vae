import torch
from torch import nn
import torch.nn.functional as F


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


# TODO: pixel unshuffle?!


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(1, 9), padding=(1, 4))
        self.bn1 = nn.BatchNorm2d(10)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(10, 10, kernel_size=(1, 9), padding=(1, 4))
        self.bn2 = nn.BatchNorm2d(10)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(10, 10, kernel_size=(1, 9), padding=(1, 4))
        self.bn3 = nn.BatchNorm2d(10)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = F.max_pool2d(x, (1, 2))

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = F.max_pool2d(x, (1, 2))

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = F.max_pool2d(x, (1, 2))

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(10, 20, kernel_size=(1, 9), padding=(1, 4))
        self.bn1 = nn.BatchNorm2d(20)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(10, 20, kernel_size=(1, 9), padding=(1, 4))
        self.bn2 = nn.BatchNorm2d(20)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(10, 20, kernel_size=(1, 9), padding=(1, 4))
        self.c1x1 = nn.Conv2d(10, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = pixel_shuffle(x, 1, 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = pixel_shuffle(x, 1, 2)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = pixel_shuffle(x, 1, 2)

        return self.c1x1(x)
