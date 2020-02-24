import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightConv2d(nn.Module):
    """
    Lightweight Convolution assuming the input is BxCxHxW
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution
    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    """

    def __init__(self, input_size, kernel_size=1, padding=0, num_heads=1,
                 weight_softmax=False, bias=False, weight_dropout=0.):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None
        self.weight_dropout = weight_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self, input):
        """
        input size: B x C x H x W
        output size: B x C x H x W
        """
        B, C, H, W = input.size()
        N = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        weight = F.dropout(weight, self.weight_dropout, training=self.training)
        # Merge every C/H entries into the batch dimension (C = self.input_size)
        # B x C x T -> (B * C/H) x H x T
        # One can also expand the weight to C x 1 x K by a factor of C/H
        # and do not reshape the input instead, which is slow though
        input = input.view(-1, N, H, W)
        output = F.conv1d(input, weight, padding=self.padding, groups=self.num_heads)
        output = output.view(B, C, H, W)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output
