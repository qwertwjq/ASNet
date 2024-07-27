
import math

import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
            input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True,
           this module has learnable per-element affine parameters
           initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.dim, 1))
            self.bias = nn.Parameter(torch.zeros(self.dim, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x L
        # gln: mean,var N x 1 x 1
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))

        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
        # N x C x L
        if self.elementwise_affine:
            x = self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine)

    def forward(self, x):
        # x: N x C x L
        # N x L x C
        x = torch.transpose(x, 1, 2)
        # N x L x C == only channel norm
        x = super().forward(x)
        # N x C x L
        x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim):
    # if norm not in ['gln', 'cln', 'bn']:
    # if x.dim() != 3:
    #     raise RuntimeError("{} accept 3D tensor as input".format(
    #         self.__name__))

    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    else:
        return nn.BatchNorm1d(dim)


class Basic(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding=0, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        groups = 1
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
class ChannelPool(nn.Module):
    def __init__(self):
        super(ChannelPool, self).__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SAB(nn.Module):
    def __init__(self):
        super(SAB, self).__init__()
        kernel_size = 5
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale
class CABlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., norm='cln'):
        super().__init__()

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv1d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sab = SAB()
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv1d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv1d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = select_norm(norm, c)
        self.norm2 = select_norm(norm, c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.sab(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    '''
       This module can be seen as the gradient of Conv1d with respect to its input.
       It is also known as a fractionally-strided convolution
       or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x

# class eca_block(nn.Module):
#     # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
#     def __init__(self, in_channel, b=1, gama=2):
#         # 继承父类初始化
#         super(eca_block, self).__init__()
#
#         # 根据输入通道数自适应调整卷积核大小
#         kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))
#         # 如果卷积核大小是奇数，就使用它
#         if kernel_size % 2:
#             kernel_size = kernel_size
#         # 如果卷积核大小是偶数，就把它变成奇数
#         else:
#             kernel_size = kernel_size-1
#         # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
#         padding = kernel_size // 2
#
#         # 全局平均池化，输出的特征图的宽高=1
#         self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
#         # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
#         self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
#                               bias=False, padding=padding)
#         # sigmoid激活函数，权值归一化
#         self.sigmoid = nn.Sigmoid()
#
#     # 前向传播
#     def forward(self, inputs):
#         # 获得输入图像的shape
#         b, c, s = inputs.shape
#
#         x = self.avg_pool(inputs)
#         x = x.view([b, 1, c])
#
#         x = self.conv(x)
#         # 权值归一化
#         x = self.sigmoid(x)
#         x = x.view([b, c, 1])
#         outputs = x * inputs
#         return outputs
class DCBlock(nn.Module):
    '''
       Consider only residual links
    '''

    def __init__(self, in_channels=16, out_channels=32,
                 kernel_size=3, dilation=1, norm='cln',drop_out_rate = 0.5):
        super(DCBlock, self).__init__()
        # conv 1 x 1
        self.conv1x1 = Conv1D(in_channels, out_channels, 1)
        self.PReLU_1 = nn.PReLU()
        self.norm_1 = select_norm(norm, out_channels)
        self.pad = (dilation * (kernel_size - 1)) // 2
        # depthwise convolution
        self.dwconv = Conv1D(out_channels, out_channels, kernel_size,
                             groups=out_channels, padding=self.pad, dilation=dilation)
        self.PReLU_2 = nn.PReLU()
        self.norm_2 = select_norm(norm, out_channels)
        self.Sc_conv = nn.Conv1d(out_channels, in_channels, 1, bias=True)


    def forward(self, x):

        c = self.conv1x1(x)

        # N x O_C x L
        c = self.PReLU_1(c)
        c = self.norm_1(c)
        # causal: N x O_C x (L+pad)
        # noncausal: N x O_C x L
        c = self.dwconv(c)
        # N x O_C x L
        c = self.Sc_conv(c)
        return x + c
num  = 8
# 0.55mb
num_r = 8
class ASNet(nn.Module):

    def __init__(self,
                 seq_channel=1,
                 width=8,
                 enc_blk_nums=[num, num, num],
                 dec_blk_nums=[num, num, num],
                 B = 64,
                 H = 64,
                 P = 3,
                 X = 6,
                 R = num_r,
                 norm = "cln",

    ):
        super().__init__()

        self.intro = nn.Conv1d(in_channels=seq_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv1d(in_channels=width, out_channels=seq_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)   #bias因为一般为False的时候，nn.Conv2d()后面通常接nn.BatchNorm2d(output)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.AvgPool = nn.AdaptiveAvgPool1d(1)
        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[CABlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv1d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.TCBs = self._Sequential_repeat(
            R, X, in_channels=B, out_channels=H, kernel_size=P, norm=norm)

        self.gen_masks = nn.Conv1d(in_channels=B, out_channels=chan, kernel_size=1, stride=1)


        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose1d(chan, chan // 2, 2, 2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[CABlock(chan) for _ in range(num)]
                )
            )

        self.PReLU_2 = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
    def TC_block(self, num_blocks, **block_kwargs):

        DCBlocks = [DCBlock(
            **block_kwargs, dilation=(2**i)) for i in range(num_blocks)]

        return nn.Sequential(*DCBlocks)

    def _Sequential_repeat(self, num_repeats, num_blocks, **block_kwargs):

        TC_blocks = [self.TC_block(
            num_blocks, **block_kwargs) for i in range(num_repeats)]
        return nn.Sequential(*TC_blocks)

    def forward(self, inp):
        inp = torch.unsqueeze(inp, 1)
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        print(x.shape)
        e = self.TCBs(x)

        e = self.PReLU_2(e)
        m = self.gen_masks(e)
        fl_x_s = self.sigmoid(m)
        x = x*fl_x_s

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):  # [start : end: step]  a[ : : -1]可实现逆序
            x = up(x)
            x = x + enc_skip

            x = decoder(x)

        x = self.ending(x)
        x = torch.squeeze(x, 1)
        return x



def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10 ** 6


if __name__ == '__main__':


    net = ASNet()
    print(str(check_parameters(net)) + ' Mb')

    test = torch.randn(10, 512)
    zhi = net(test)
    print(zhi.shape)



