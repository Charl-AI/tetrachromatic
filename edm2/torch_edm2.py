"""Reference PyTorch implementation for EDM2 (Karras et al.).
'Analyzing and Improving the Training Dynamics of Diffusion Models'.
Forked from github.com/NVlabs/edm2 (CC BY-NC-SA 4.0 license)."""

import os
import tempfile
from functools import partial

import numpy as np
import torch

# fmt:off

# Cache directories
# ------------------------------------------------------------------------------------------

_dnnlib_cache_dir = None


def set_cache_dir(path: str) -> None:
    global _dnnlib_cache_dir
    _dnnlib_cache_dir = path


def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if "DNNLIB_CACHE_DIR" in os.environ:
        return os.path.join(os.environ["DNNLIB_CACHE_DIR"], *paths)
    if "HOME" in os.environ:
        return os.path.join(os.environ["HOME"], ".cache", "dnnlib", *paths)
    if "USERPROFILE" in os.environ:
        return os.path.join(os.environ["USERPROFILE"], ".cache", "dnnlib", *paths)
    return os.path.join(tempfile.gettempdir(), ".cache", "dnnlib", *paths)


# ----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (
        value.shape,
        value.dtype,
        value.tobytes(),
        shape,
        dtype,
        device,
        memory_format,
    )
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


# ----------------------------------------------------------------------------
# Variant of constant() that inherits dtype and device from the given
# reference tensor by default.


def const_like(ref, value, shape=None, dtype=None, device=None, memory_format=None):
    if dtype is None:
        dtype = ref.dtype
    if device is None:
        device = ref.device
    return constant(
        value, shape=shape, dtype=dtype, device=device, memory_format=memory_format
    )



#----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.

def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

#----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.

def resample(x, f=[1,1], mode='keep'):
    if mode == 'keep':
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = const_like(x, f)
    c = x.shape[1]
    if mode == 'down':
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))
    assert mode == 'up'
    return torch.nn.functional.conv_transpose2d(x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,))

#----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).

def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596

#----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).

def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t ** 2)

#----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).

def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t ** 2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a , wb * b], dim=dim)

#----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).

class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).

class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w)) # forced weight normalization
        w = normalize(w) # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel())) # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=(w.shape[-1]//2,))

#----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).

class Block(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        emb_channels,                   # Number of embedding channels.
        flavor              = 'enc',    # Flavor: 'enc' or 'dec'.
        resample_mode       = 'keep',   # Resampling: 'keep', 'up', or 'down'.
        resample_filter     = [1,1],    # Resampling filter.
        attention           = False,    # Include self-attention?
        channels_per_head   = 64,       # Number of channels per attention head.
        dropout             = 0,        # Dropout probability.
        res_balance         = 0.3,      # Balance between main branch (0) and residual branch (1).
        attn_balance        = 0.3,      # Balance between main branch (0) and self-attention (1).
        clip_act            = 256,      # Clip output activations. None = do not clip.
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.emb_gain = torch.nn.Parameter(torch.zeros([]))
        self.conv_res0 = MPConv(out_channels if flavor == 'enc' else in_channels, out_channels, kernel=[3,3])
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3,3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1,1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1,1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1,1]) if self.num_heads != 0 else None

    def forward(self, x, emb):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1) # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c.unsqueeze(2).unsqueeze(3).to(y.dtype))
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == 'dec' and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            y = self.attn_qkv(x) # type: ignore
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3) # pixel norm & split
            w = torch.einsum('nhcq,nhck->nhqk', q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum('nhqk,nhck->nhcq', w, v)
            y = self.attn_proj(y.reshape(*x.shape)) # type: ignore
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x

#----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).

class UNet(torch.nn.Module):
    def __init__(self, *,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        label_dim,                          # Class label dimensionality. 0 = unconditional.
        output_mult         = 1,            # How many output channels to predict (as a multiplier of img_channels)
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16,8],       # List of resolutions with self-attention.
        label_balance       = 0.5,          # Balance between noise embedding (0) and class embedding (1).
        concat_balance      = 0.5,          # Balance between skip connections (0) and main path (1).
        **block_kwargs,                     # Arguments for Block.
    ):
        super().__init__()
        cblock = [model_channels * x for x in channel_mult]
        cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        self.label_dim = label_dim

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(label_dim, cemb, kernel=[]) if label_dim != 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = img_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_conv'] = MPConv(cin, cout, kernel=[3,3])
            else:
                self.enc[f'{res}x{res}_down'] = Block(cout, cout, cemb, flavor='enc', resample_mode='down', **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='enc', attention=(res in attn_resolutions), **block_kwargs)

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            if level == len(cblock) - 1:
                self.dec[f'{res}x{res}_in0'] = Block(cout, cout, cemb, flavor='dec', attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = Block(cout, cout, cemb, flavor='dec', **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = Block(cout, cout, cemb, flavor='dec', resample_mode='up', **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                self.dec[f'{res}x{res}_block{idx}'] = Block(cin, cout, cemb, flavor='dec', attention=(res in attn_resolutions), **block_kwargs)
        self.out_conv = MPConv(cout, img_channels * output_mult, kernel=[3,3])

    def forward(self, 
        x, # torch.float32 (B,C,H,W) range [-1,1]
        t, # torch.float32 (B,)      range [0,1]
        y, # torch.float32 (B)       range [0, label_dim)
    ):
        y = torch.nn.functional.one_hot(y, num_classes=self.label_dim).to(t.dtype)

        emb = self.emb_noise(self.emb_fourier(t))
        if self.emb_label is not None:
            emb = mp_sum(emb, self.emb_label(y * np.sqrt(y.shape[1])), t=self.label_balance)
        emb = mp_silu(emb)

        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x

# In the edm2 paper, all models were trained on 32 GPUs
# with a batch size of 64 per GPU (global batch size = 2048).
# training cost comments are for the unconditional ImageNet64 models reported in the paper
# notice that the XL model trains fastest (apart from the S variant) due to converging in fewer steps

EDM2S = partial(UNet, model_channels=192, dropout=0.0)   # ~525k steps, 2.6 days, 101.9 GFLOPS, 280M params
EDM2M = partial(UNet, model_channels=256, dropout=0.1)   # ~1.0M steps, 7.8 days, 180.8 GFLOPS, 498M params
EDM2L = partial(UNet, model_channels=320, dropout=0.1)   # ~525k steps, 5.8 days, 282.1 GFLOPS, 777M params
EDM2XL = partial(UNet, model_channels=384, dropout=0.1)  # ~325k steps, 4.9 days, 405.9 GFLOPS, 1.1B params
