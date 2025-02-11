import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import spectral as spy

from einops import rearrange
import math
import warnings
from torch import einsum

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def concat_img(psf_tmp, batch_num, channel, img_size):
        psf_tmp = torch.reshape(psf_tmp, (9, 10, channel, img_size, img_size))
        psf_tmp = psf_tmp.transpose(4, 2).transpose(3, 2)
        psf_tmp = psf_tmp.transpose(2, 1)
        psf_tmp = torch.reshape(psf_tmp, (9 * img_size, 10 * img_size, channel))
        psf_tmp = psf_tmp.transpose(2, 0).transpose(2, 1).unsqueeze(0)
        psf_tmp = torch.tile(psf_tmp, (batch_num, 1, 1, 1))

        return psf_tmp

class BatchNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x, *args, **kwargs):
        x = x.permute(0, 3, 1, 2)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        return self.fn(x, *args, **kwargs)

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class HS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(16, 16),
            dim_head=16,
            heads=8,
            only_local_branch=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.only_local_branch = False

        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, dim)

        h, w = 576 // self.heads, 640 // self.heads 
        self.global_atten = Attention(dim,inner_dim, num_patches=[h//window_size[0],w//window_size[1]], num_heads=heads,)

        self.prompt_conv1 = DoubleConv(32, 16)
        self.prompt_conv2 = DoubleConv(64, 32)
        self.prompt_conv3 = DoubleConv(128, 64)

    def forward(self, x, pmt, dim):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        h_num,w_num = (h//w_size[0]) , (w//w_size[1])
        hw_num = h_num * w_num
        assert h % w_size[0] == 0 and w % w_size[1] == 0, f'fmap dimensions must be divisible by the window size {h} {w}'

        if dim == 16:
            xpmt = torch.cat([x.permute(0, 3, 1, 2), pmt], 1)
            xpmt = self.prompt_conv1(xpmt).permute(0, 2, 3, 1)
        elif dim == 32:
            xpmt = torch.cat([x.permute(0, 3, 1, 2), pmt], 1)
            xpmt = self.prompt_conv2(xpmt).permute(0, 2, 3, 1)
        elif dim == 64:
            xpmt = torch.cat([x.permute(0, 3, 1, 2), pmt], 1)
            xpmt = self.prompt_conv3(xpmt).permute(0, 2, 3, 1)
        else:
            xpmt = x

        v = self.to_v(xpmt)
        k = self.to_k(xpmt)
        q = self.to_q(xpmt)
        
        q, k, v = map(lambda t: rearrange(t, 'b (h b0) (w b1) c -> b (h w) (c b0 b1)',
                                                b0=w_size[0], b1=w_size[1]), (q, k, v))

        out = self.global_atten(q, k, v)
        out = rearrange(out, 'b (h w) (c b0 b1) -> b (h b0) (w b1) c', b0=w_size[0], b1=w_size[1], h=h_num,
                              w=w_num)
        out = self.to_out(out)

        return out

class Attention(nn.Module):
    def __init__(self, dim,inner_dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.attention_mask = np.load('./params/attention_mask.npy' ,allow_pickle=True).item()
        self.mask90 = torch.tensor(self.attention_mask['mask90']).cuda()
        self.mask360 = torch.tensor(self.attention_mask['mask360']).cuda()
        self.mask1440 = torch.tensor(self.attention_mask['mask1440']).cuda()

        self.dim = dim
        self.inner_dim = inner_dim
        self.num_patches = num_patches
        self.num_heads = num_heads
        head_dim = inner_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        seq_l = num_patches[0]*num_patches[1]
        self.positional_encoding = nn.Parameter(torch.Tensor(1, num_heads, seq_l, seq_l))
        trunc_normal_(self.positional_encoding)

    def forward(self, q, k, v, H=None, W=None):

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.positional_encoding
        
        # FoV guidance
        if sim.shape[2] == 90:
            sim = sim * self.mask90
        elif sim.shape[2] == 360:
            sim = sim * self.mask360
        elif sim.shape[2] == 1440:
            sim = sim * self.mask1440
        else:
            raise NotImplementedError

        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b  n (h d)', h=self.num_heads)

        return out


class HSAB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(16, 16),
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                BatchNorm(dim, HS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, only_local_branch=(heads==1))),
                BatchNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, pmt, dim):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, pmt, dim) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

class DRT(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=16, num_blocks=[1,1,1]):
        super(DRT, self).__init__()
        self.dim = dim
        self.scales = len(num_blocks)

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scales-1):
            self.encoder_layers.append(nn.ModuleList([
                HSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
            ]))
            dim_scale *= 2

        # Bottleneck
        self.bottleneck = HSAB(dim=dim_scale, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_scale, dim_scale // 2, 1, 1, bias=False),
                HSAB(dim=dim_scale // 2, num_blocks=num_blocks[self.scales - 2 - i], dim_head=dim,
                     heads=(dim_scale // 2) // dim),
            ]))
            dim_scale //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # Activation function
        self.apply(self._init_weights)

        # PSF features
        self.promptin = (DoubleConv(7, 16))
        self.promptdown1 = (Down(16, 32))
        self.promptdown2 = (Down(32, 64))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, psfs):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # Embedding
        fea = self.embedding(x)

        # PSF Prompt
        pad_size = 10
        psfs_pad = F.pad(psfs, pad=(pad_size, pad_size, pad_size, pad_size, 0, 0), mode='constant')
        pmt_x1 = self.promptin(psfs_pad)
        pmt_x1_input = concat_img(pmt_x1, b, pmt_x1.shape[1], pmt_x1.shape[2])
        pmt_x2 = self.promptdown1(pmt_x1)
        pmt_x2_input = concat_img(pmt_x2, b, pmt_x2.shape[1], pmt_x2.shape[2])
        pmt_x3 = self.promptdown2(pmt_x2)
        pmt_x3_input = concat_img(pmt_x3, b, pmt_x3.shape[1], pmt_x3.shape[2])

        # Encoder
        fea_encoder = []
        for (HSAB, FeaDownSample) in self.encoder_layers:
            if fea.shape[1] == 16:
                fea = HSAB(fea, pmt_x1_input, 16)
                fea_encoder.append(fea)
                fea = FeaDownSample(fea)
            elif fea.shape[1] == 32:
                fea = HSAB(fea, pmt_x2_input, 32)
                fea_encoder.append(fea)
                fea = FeaDownSample(fea)
            else:
                raise NotImplementedError

        # Bottleneck
        fea = self.bottleneck(fea, pmt_x3_input, 64)

        # Decoder
        for i, (FeaUpSample, Fution, HSAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.scales-2-i]], dim=1))
            fea = HSAB(fea, pmt_x2_input, 0)  

        # Mapping
        out = self.mapping(fea) + x
        return out[:, :, :h_inp, :w_inp]
    
def ReconModel():
    return DRT() 