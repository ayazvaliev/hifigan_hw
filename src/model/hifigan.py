import torch
import torch.nn as nn
import torch.nn.functional as F


class MRFBlock(nn.Module):
    def __init__(self, channel_dim, kernel_sizes, dilations_tuple, negative_slope):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Conv1d(
                            in_channels=channel_dim,
                            out_channels=channel_dim,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=(kernel_size - 1) * dilation // 2,
                        )
                        for dilation in dilations
                    ]
                )
                for kernel_size, dilations in zip(kernel_sizes, dilations_tuple)
            ]
            
        )
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, x: torch.Tensor):
        res = torch.zeros_like(x, requires_grad=True)
        for block in self.blocks:
            cur_x = x.clone()
            for conv_layer in block:
                cur_x += conv_layer(self.act(cur_x))
            res += cur_x
        return res / len(self.blocks)


class Generator(nn.Module):
    def __init__(self,
                 n_bands,
                 hidden_dim,
                 proj_kernel_size,
                 upsample_kernel_sizes,
                 mrf_kernel_sizes,
                 mrf_dilations,
                 negative_slope,
                 ):
        super().__init__()
        self.conv_proj = nn.Conv1d(
            in_channels=n_bands,
            out_channels=hidden_dim,
            kernel_size=proj_kernel_size,
            padding=proj_kernel_size // 2
        )
        self.gen_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels= hidden_dim // 2**i,
                        out_channels= hidden_dim // 2**(i+1),
                        stride=upsample_kernel_size // 2,
                        kernel_size=upsample_kernel_size,
                        padding=upsample_kernel_size // 4
                    ),
                    MRFBlock(
                        channel_dim=hidden_dim // 2**(i+1),
                        kernel_sizes=mrf_kernel_sizes,
                        dilations_tuple=mrf_dilations,
                        negative_slope=negative_slope,
                    )
                )
                for i, upsample_kernel_size in enumerate(upsample_kernel_sizes)
            ]
        )
        self.convert_to_wav = nn.Conv1d(
            in_channels=hidden_dim // 2**(len(upsample_kernel_sizes)),
            out_channels=1,
            kernel_size=proj_kernel_size,
            padding=proj_kernel_size // 2)
        
        self.act = nn.LeakyReLU(negative_slope)
        self.final_act = torch.nn.Tanh()

    def forward(self, x: torch.Tensor):
        # x [N, F, T]
        x = self.conv_proj(x)
        for block in self.gen_blocks:
            x = block(self.act(x))
        return self.final_act(self.convert_to_wav(self.act(x))) # [N, 1, T]

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

class MPD(nn.Module):
    def __init__(self,
                 periods,
                 num_layers,
                 negative_slope):
        super().__init__()
        self.periods = periods
        self.discriminators = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Sequential(
                        nn.Conv2d(
                            in_channels=1,
                            out_channels=2**6,
                            kernel_size=(5, 1),
                            stride=(3, 1),
                            padding=(5 // 2, 0)
                        ),
                        nn.LeakyReLU(negative_slope)
                    )]
                    +
                    [nn.Sequential(
                        nn.Conv2d(
                            in_channels=2**(5 + i),
                            out_channels=2**(6 + i),
                            kernel_size=(5, 1),
                            stride=(3, 1),
                            padding=(5 // 2, 0)
                        ),
                        nn.LeakyReLU(negative_slope)
                    ) for i in range(1, num_layers)]
                    +
                    [nn.Sequential(
                        nn.Conv2d(
                            in_channels=2**(5 + num_layers),
                            out_channels=1024,
                            kernel_size=(5, 1),
                            padding=(5 // 2, 0)
                        ),
                        nn.LeakyReLU(negative_slope),
                        nn.Conv2d(
                            in_channels=1024,
                            out_channels=1,
                            kernel_size=(3, 1),
                            padding=(3 // 2, 0)
                        )
                    )]
                )
            for _ in range(len(periods))
            ]
        )

    def expand_to_period(self, x: torch.Tensor, period):
        B, C, T = x.shape
        if T % period:
            x = F.pad(x, pad=(0, (T // period + 1) * period - T), mode='reflect')
        return x.view(B, C, -1, period)

    def forward(self, x: torch.Tensor, gt: torch.Tensor):
        gt = gt.unsqueeze(1)

        latents_per_period = []
        latents_per_period_gt = []
        for period, discriminator in zip(self.periods, self.discriminators):
            cur_latents = []
            cur_latents_gt = []
            
            cur_x = self.expand_to_period(x, period)
            cur_gt = self.expand_to_period(gt, period)
            
            for conv_layer in discriminator:
                cur_x = conv_layer(cur_x)
                cur_latents.append(cur_x)

                cur_gt = conv_layer(cur_gt)
                cur_latents_gt.append(cur_gt)

            latents_per_period.append(cur_latents)
            latents_per_period_gt.append(cur_latents_gt)

        return latents_per_period, latents_per_period_gt


class Conv1dAct(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride,
                 groups, 
                 negative_slope = 0,
                 use_act=True):
        super().__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups
        )
        self.act = nn.LeakyReLU(negative_slope) if use_act else nn.Identity()
    
    def forward(self, x: torch.Tensor):
        return self.act(self.conv_layer(x))

class MSD(nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Conv1dAct(
                            in_channels=1,
                            out_channels=16,
                            kernel_size=15,
                            stride=1,
                            groups=1,
                            negative_slope=negative_slope
                        ),
                        Conv1dAct(
                            in_channels=16,
                            out_channels=64,
                            kernel_size=41,
                            stride=4,
                            groups=4,
                            negative_slope=negative_slope
                        ),
                        Conv1dAct(
                            in_channels=64,
                            out_channels=256,
                            kernel_size=41,
                            stride=4,
                            groups=16,
                            negative_slope=negative_slope
                        ),
                        Conv1dAct(
                            in_channels=256,
                            out_channels=1024,
                            kernel_size=41,
                            stride=4,
                            groups=64,
                            negative_slope=negative_slope
                        ),
                        Conv1dAct(
                            in_channels=1024,
                            out_channels=1024,
                            kernel_size=41,
                            stride=4,
                            groups=256,
                            negative_slope=negative_slope
                        ),
                        Conv1dAct(
                            in_channels=1024,
                            out_channels=1024,
                            kernel_size=5,
                            stride=1,
                            groups=1,
                            negative_slope=negative_slope
                        ),
                        Conv1dAct(
                            in_channels=1024,
                            out_channels=1,
                            kernel_size=3,
                            stride=1,
                            groups=1,
                            use_act=False
                        )
                    ]
                )
                for _ in range(3)
            ]
        )
        self.avg_pools = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, 2),
            nn.AvgPool1d(4, 2, 2)
        ])
    
    def forward(self, x: torch.Tensor, gt: torch.Tensor):
        latents_per_period = []
        latents_per_period_gt = []

        for discriminator, avg_pool in zip(self.discriminators, self.avg_pools):
            cur_latents = []
            cur_latents_gt = []
            
            x = avg_pool(x)
            gt = avg_pool(gt)

            cur_x = x.clone()
            cur_gt = gt.clone()

            for conv_layer in discriminator:
                cur_x = conv_layer(cur_x)
                cur_latents.append(cur_x)

                cur_gt = conv_layer(cur_gt)
                cur_latents_gt.append(cur_gt)

            latents_per_period.append(cur_latents)
            latents_per_period_gt.append(cur_latents_gt)

        return latents_per_period, latents_per_period_gt


class HifiGAN(nn.Module):
    def __init__(self,
                 melspec_transformer,
                 n_bands,
                 gen_hidden_dim,
                 gen_proj_kernel_size,
                 gen_upsample_kernel_sizes,
                 mrf_kernel_sizes,
                 mrf_dilations,
                 mpd_periods,
                 mpd_num_layers,
                 negative_slope,
                 ):
        super().__init__()
        self.melspec_transformer = melspec_transformer
        self.generator = Generator(
            n_bands=n_bands,
            hidden_dim=gen_hidden_dim,
            proj_kernel_size=gen_proj_kernel_size,
            upsample_kernel_sizes=gen_upsample_kernel_sizes,
            mrf_kernel_sizes=mrf_kernel_sizes,
            mrf_dilations=mrf_dilations,
            negative_slope=negative_slope
        )
        self.mpd = MPD(
            periods=mpd_periods,
            num_layers=mpd_num_layers,
            negative_slope=negative_slope
        )
        self.msd = MSD(
            negative_slope=negative_slope
        )

    def switch_grad_mode(self, module: nn.Module, state=True):
        for p in module.parameters():
            p.requires_grad = state
    
    def forward_discriminator(self, spectrogram, audio, **kwargs):
        self.switch_grad_mode(self.mpd, True)
        self.switch_grad_mode(self.msd, True)

        generated = self.generator(spectrogram)
        if generated.size(-1) > audio.size(-1):
            generated = generated[..., :audio.size(-1)]
        elif generated.size(-1) < audio.size(-1):
            audio = audio[..., :generated.size(-1)]

        mpd_latent_per_period, mpd_latent_per_period_gt = self.mpd(generated.detach(), audio)
        msd_latent_per_period, msd_latent_per_period_gt = self.msd(generated.detach(), audio)

        return {
            "generated": generated,
            "mpd_latent_per_period": mpd_latent_per_period,
            "mpd_latent_per_period_gt": mpd_latent_per_period_gt,
            "msd_latent_per_period": msd_latent_per_period,
            "msd_latent_per_period_gt": msd_latent_per_period_gt,
        }

    def forward_generator(self, generated, audio, **kwargs):
        self.switch_grad_mode(self.mpd, False)
        self.switch_grad_mode(self.msd, False)

        mpd_latent_per_period, mpd_latent_per_period_gt = self.mpd(generated, audio)
        msd_latent_per_period, msd_latent_per_period_gt = self.msd(generated, audio)

        return {
            "mpd_latent_per_period": mpd_latent_per_period,
            "mpd_latent_per_period_gt": mpd_latent_per_period_gt,
            "msd_latent_per_period": msd_latent_per_period,
            "msd_latent_per_period_gt": msd_latent_per_period_gt,
            "generated_spectrogram": self.melspec_transformer(generated.squeeze(1))
        }
    
    def forward(self, x: torch.Tensor, **kwargs):
        pass

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
