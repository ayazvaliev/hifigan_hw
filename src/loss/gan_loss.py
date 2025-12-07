import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, generated_list: list[torch.Tensor], real_list: list[torch.Tensor]):
        loss = 0
        for generated, real in zip(generated_list, real_list):
            loss += torch.mean((1 - real)**2) + torch.mean(generated**2)
        return loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, generated_list: list[torch.Tensor]):
        loss = 0
        for generated in generated_list:
            loss += torch.mean((1 - generated)**2)
        return loss


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, latents_per_period: list[torch.Tensor], latents_per_period_gt: list[torch.Tensor]):
        loss = 0
        for cur_latents, cur_latents_gt in zip(latents_per_period, latents_per_period_gt):
            for latent, latent_gt in zip(cur_latents, cur_latents_gt):
                loss += torch.mean(torch.abs(latent - latent_gt)) / torch.prod(torch.tensor(latent.shape[1:]))

        return loss


class MelSpecLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, generated_spectrogram: torch.Tensor, real_spectrogram: torch.Tensor):
        if generated_spectrogram.size(-1) < real_spectrogram.size(-1):
            real_spectrogram = generated_spectrogram[..., :generated_spectrogram.size(-1)]
        
        return torch.mean(torch.abs(generated_spectrogram - real_spectrogram))


class HiFiGANDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc_loss = DiscriminatorLoss()
    
    def forward(self, 
                mpd_latent_per_period, 
                msd_latent_per_period, 
                mpd_latent_per_period_gt,
                msd_latent_per_period_gt,
                **kwargs):
        
        mpd_disc_loss = self.disc_loss(mpd_latent_per_period[-1], mpd_latent_per_period_gt[-1])
        msd_disc_loss = self.disc_loss(msd_latent_per_period[-1], msd_latent_per_period_gt[-1])
        return {"disc_loss": mpd_disc_loss + msd_disc_loss}


class HiFiGANGeneratorLoss(nn.Module):
    def __init__(self, melspec_loss_coeff, feature_loss_coeff):
        super().__init__()
        self.melspec_loss_coeff = melspec_loss_coeff
        self.feature_loss_coeff = feature_loss_coeff

        self.gen_loss = GeneratorLoss()
        self.feature_loss = FeatureLoss()
        self.melspec_loss = MelSpecLoss()
    
    def forward(self, 
                mpd_latent_per_period,
                msd_latent_per_period,
                mpd_latent_per_period_gt,
                msd_latent_per_period_gt,
                generated_spectrogram,
                spectrogram,
                **kwargs
                ):
        mpd_gen_loss = self.gen_loss(mpd_latent_per_period[-1])
        msd_gen_loss = self.gen_loss(msd_latent_per_period[-1])
        mpd_feature_loss = self.feature_loss(mpd_latent_per_period, mpd_latent_per_period_gt)
        msd_features_loss = self.feature_loss(msd_latent_per_period, msd_latent_per_period_gt)
        melspec_loss = self.melspec_loss(generated_spectrogram, spectrogram)

        total_gen_loss = mpd_gen_loss + msd_gen_loss + self.feature_loss_coeff * (mpd_feature_loss + msd_features_loss) + self.melspec_loss_coeff * melspec_loss
        return {"gen_loss": total_gen_loss, "melspec_loss": melspec_loss}

    