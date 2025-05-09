# Copyright (c) 2022 NVIDIA CORPORATION.
#   Licensed under the MIT license.
# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import logging
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.nn.utils.weight_norm import WeightNorm

from .activations import SnakeBeta
from .alias_free_torch import *



LRELU_SLOPE = 0.1

logger = logging.getLogger(__name__)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class AMPBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(AMPBlock1, self).__init__()

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers

        self.activations = nn.ModuleList([
            Activation1d(activation=SnakeBeta(channels, alpha_logscale=True))
            for _ in range(self.num_layers)
        ])

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def set_weight_norm(self, enabled: bool):
        weight_norm_fn = weight_norm if enabled else remove_weight_norm
        for l in self.convs1:
            weight_norm_fn(l)
        for l in self.convs2:
            weight_norm_fn(l)


class BigVGAN(nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.

    # We've got a model in prod that has the wrong hparams for this. It's simpler to add this check than to
    # redistribute the model.
    ignore_state_dict_unexpected = ("cond_layer.*",)

    def __init__(self):
        super().__init__()

        input_dims = 80

        upsample_rates = [10, 8, 4, 2]
        upsample_kernel_sizes = [x * 2 for x in upsample_rates]
        upsample_initial_channel = 1024

        resblock_kernel_sizes = [3, 7, 11]
        resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        # pre conv
        self.conv_pre = weight_norm(Conv1d(input_dims, upsample_initial_channel, 7, 1, padding=3))
        self.cond_layer = None

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                            upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(AMPBlock1(ch, k, d))

        # post conv
        activation_post = SnakeBeta(ch, alpha_logscale=True)
        self.activation_post = Activation1d(activation=activation_post)
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x) -> torch.Tensor:
        """
        Args
        ----
        x: torch.Tensor of shape [B, T, C]
        """
        with torch.inference_mode():

            x = self.conv_pre(x)

            for i in range(self.num_upsamples):
                # upsampling
                for i_up in range(len(self.ups[i])):
                    x = self.ups[i][i_up](x)
                # AMP blocks
                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.resblocks[i * self.num_kernels + j](x)
                    else:
                        xs += self.resblocks[i * self.num_kernels + j](x)
                x = xs / self.num_kernels

            # post conv
            x = self.activation_post(x)
            x = self.conv_post(x)

            # Bound the output to [-1, 1]
            x = torch.tanh(x)

            return x

    @property
    def weight_norm_enabled(self) -> bool:
        return any(
            isinstance(hook, WeightNorm) and hook.name == "weight"
            for k, hook in self.conv_pre._forward_pre_hooks.items()
        )

    def set_weight_norm(self, enabled: bool):
        """
        N.B.: weight norm modifies the state dict, causing incompatibilities. Conventions:
        - BigVGAN runs with weight norm for training, without for inference (done automatically by instantiate())
        - All checkpoints are saved with weight norm (allows resuming training)
        """
        if enabled != self.weight_norm_enabled:
            weight_norm_fn = weight_norm if enabled else remove_weight_norm
            logger.debug(f"{'Applying' if enabled else 'Removing'} weight norm...")

            for l in self.ups:
                for l_i in l:
                    weight_norm_fn(l_i)
            for l in self.resblocks:
                l.set_weight_norm(enabled)
            weight_norm_fn(self.conv_pre)
            weight_norm_fn(self.conv_post)

    def train_mode(self):
        self.train()
        self.set_weight_norm(enabled=True)

    def inference_mode(self):
        self.eval()
        self.set_weight_norm(enabled=False)


if __name__ == '__main__':
    import sys
    import soundfile as sf
    model = BigVGAN()

    state_dict = torch.load("bigvgan32k.pt")
    msg = model.load_state_dict(state_dict)
    model.eval()
    model.set_weight_norm(enabled=False)

    print(msg)
    mels = torch.load("mels.pt")
    with torch.inference_mode():
        y = model(mels.cpu())

    for i, wav in enumerate(y):
        wav = wav.view(-1).detach().numpy()
        sf.write(f"bigvgan_test{i}.flac", wav, samplerate=32_000, format="FLAC")
