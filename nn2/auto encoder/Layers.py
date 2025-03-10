import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tv
import PIL
import random
from delegation import delegates

def init_identity(c):
    with torch.no_grad():
        num = min(c.weight.shape[0], c.weight.shape[1])
        k = c.weight.shape[2]
        nn.init.constant_(c.weight[:num,:num],0)
        I = torch.arange(num)
        c.weight[I,I,k//2,k//2] = 1
        nn.init.constant_(c.bias,0)

def conv2d(ni,no=None,
           k=3,s=1,pad="same",g=1,
           bias=True,init='none'):
    if no is None: no = ni
    if pad=="same": pad = (k-1)//2
    c = nn.Conv2d(in_channels = ni,
                    out_channels = no,
                    kernel_size = k,
                    stride = s,
                    padding = pad,
                    groups = g,
                    bias = bias,)
    if init == 'identity':
        assert(s==1)
        init_identity(c) 
    return c
                
@delegates(conv2d)
def cab(ni, no=None, bn=True, 
        activation=True, act_fn=nn.ReLU(),
        bn_init_zero=False, **kwargs):
    layers = []
    layers += [conv2d(ni,no,**kwargs)]
    no = layers[0].out_channels
    if activation:
        layers += [act_fn]
    if bn:
        layers += [nn.BatchNorm2d(no)]
    if bn_init_zero:
        nn.init.constant_(layers[-1].weight,0)
    return layers

class ResBlock(nn.Module):
    def __init__(self, ni, no=None, bottle=None, activation = nn.CELU(), g=1):
        super().__init__()
        if no is None: no = ni
        if bottle is None: bottle = ((ni + no) // 4)
        self.res = nn.Sequential(
                    *cab(ni, bottle, bn=False,g=g),
                    *cab(bottle, no, bn_init_zero=True, activation=False,g=g),
                )
        if no < ni:
            self.m = conv2d(ni,no,k = 1,g=g)
        if activation is None:
            activation = lambda x: x
        
        self.act = activation
        
    def forward(self, x):
        out = self.res(x)
        bs, c, h, w = out.shape
        faltan = c - x.shape[1] # no - ni
        if faltan > 0:
            Z = torch.zeros(bs,faltan,h,w,device=x.device)
            x = torch.cat([x,Z],dim=1)
        elif faltan < 0:
            x = self.m(x)
        return self.act(x + out)
        
class SelfAttention(nn.Module):
    def __init__(self, ni):
        super(SelfAttention, self).__init__()
        self.in_channels = ni

        self.query_conv = nn.Conv2d(ni, ni // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(ni, ni // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(ni, ni, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1,ni,1,1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, c, h, w = x.size()
        N = h*w
        
        query = self.query_conv(x).view(bs, -1, N).permute(0, 2, 1)  # (B, N, C')
        key = self.key_conv(x).view(bs, -1, N)  # (B, C, N)
        value = self.value_conv(x).view(bs, -1, N)  # (B, C, N)

        
        attention = query@key  # (B, N, N)
        attention = F.softmax(attention,dim=-1)

        out = value@attention.permute(0, 2, 1)  # (B, C, N)
        out = out.view(bs, c, h, w)

        return self.gamma*out + x

class ResBlockWithTime(nn.Module):
    def __init__(self, in_chs, out_chs = None, dropout_rate=0.0, time_emb_dims=512, norm_groups=32):
        if out_chs is None: out_chs = in_chs
        super().__init__()

        self.act_fn = nn.SiLU() # o ReLU, si quieres 
        # Group 1
        self.normlize_1 = nn.GroupNorm(num_groups=norm_groups, num_channels=in_chs)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv_1 = nn.Conv2d(in_chs, out_chs,
                                kernel_size=3, stride=1, padding=1)

        # Group 2 time embedding
        self.dense_1 = nn.Linear(in_features=time_emb_dims, out_features=out_chs)

        # Group 3
        self.normlize_2 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_chs)
        self.conv_2 = nn.Conv2d(out_chs, out_chs,
                                kernel_size=3, stride=1, padding="same")
        

        self.add_embedding = nn.Sequential(self.normlize_2, self.conv_2)

        if in_chs != out_chs:
            self.match_input = nn.Conv2d(in_chs, out_chs, kernel_size=1, stride=1)
        else:
            self.match_input = nn.Identity()

    def forward(self, x, t, **kwargs):
        # group 1
        h = self.act_fn(self.normlize_1(x))
        h = self.dropout(h)
        h = self.conv_1(h)

        # timestep embedding
        emb_out = self.dense_1(self.act_fn(t))[:, :, None, None]

        h = self.add_embedding(h + emb_out)
        
        # Residual
        return h + self.match_input(x)
    
class CrossAttention(nn.Module):
    def __init__(self, channels=64, n_heads=4, condition_dim=4, dropout=0.0, **kwargs):
        super().__init__()
        self.channels = channels
        self.condition_dim = condition_dim

        self.norm = nn.LayerNorm(channels)
        self.proj_cond = nn.Linear(condition_dim, channels)
        self.mhsa = nn.MultiheadAttention(embed_dim=self.channels, num_heads=n_heads, batch_first=True, dropout=dropout)

    def forward(self, x, *args, cond=None):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).swapaxes(1, 2)  # [B, C, H, W] --> [B, C, H * W] --> [B, H*W, C]
        # condition = condition or x
        cond = self.proj_cond(cond)

        q, k, v = self.norm(x), self.norm(cond)[...,None,:], self.norm(cond)[...,None,:]
        att, _ = self.mhsa(q, k, v)  # [B, H*W, C]
        x = (x + att).swapaxes(2, 1).reshape(B, C, H, W)  # [B, H*W, C] --> [B, C, H, W]
        return x

class SinusoidalPositionEmbeddings(nn.Module):
    """Positional embedding for including time information"""
    def __init__(self, total_time_steps=1000, time_emb_dims=128, time_emb_dims_exp=512):
        super().__init__()

        half_dim = time_emb_dims // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)

        ts  = torch.arange(total_time_steps, dtype=torch.float32)

        emb = torch.unsqueeze(ts, dim=-1) * torch.unsqueeze(emb, dim=0)

        res = torch.cat((emb, emb), dim=-1)
        res[:, 0::2] = emb.sin()
        res[:, 1::2] = emb.cos()

        self.time_blocks = nn.Sequential(
            nn.Embedding.from_pretrained(res, freeze=True),
            nn.Linear(in_features=time_emb_dims, out_features=time_emb_dims_exp),
            nn.SiLU(),
            nn.Linear(in_features=time_emb_dims_exp, out_features=time_emb_dims_exp),
        )

    def forward(self, time):
        return self.time_blocks(time)

class UpSample(nn.Module):
    """Up sample by 2, then convolve"""
    def __init__(self, in_channels:int):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False))
    def forward(self, x, *args, **kwargs):
        return self.upsample(x)

class DownSample(nn.Module):
    def __init__(self, channels:int, use_conv=False):
        super().__init__()
        self.downsample = nn.AvgPool2d(2, 2)
    def forward(self, x, *args, **kwargs):
        return self.downsample(x)