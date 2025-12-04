import torch
from layers import *
import torch.nn as nn
from tqdm import trange

class UNetLevel(nn.Module):
    def __init__(self,
                 *channels: list[int],
                 next_level = None,
                 num_attention_head=None, 
                 positional_encoding_type=None,
                 time_dims = 0):
        assert len(channels) >= 2, 'UNetLevel requires at least input and output channels'
        super().__init__()
        channels = list(channels) # for modify, it can't be tuple
        channels[0] += time_dims

        self.encoder = nn.Sequential(
            *[ResConvBlock(c_in, c_out, c_out) for c_in, c_out in zip(channels[:-1], channels[1:])]
        )
        channels[0] -= time_dims # we don't need time dims for output

        if next_level is not None:
            self.next_level = nn.Sequential(
                nn.Conv2d(channels[-1], channels[-1], 3, stride=2, padding=1),
                next_level,
                nn.ConvTranspose2d(channels[-1], channels[-1], 3, stride=2, padding=1, output_padding=1)
            )
        else:
            self.next_level = None

        if positional_encoding_type == FourierPositionalEncoding2d:
            self.encoder.add_module('position encoder', FourierPositionalEncoding2d(channels[-1]))
        elif positional_encoding_type == PositionalEncoding2d:
            self.encoder.add_module('position encoder', PositionalEncoding2d())
        else:
            self.positional_encoding = None

        if num_attention_head is not None:
            self.encoder.add_module('self attention', SelfAttention(channels[-1], num_attention_head))
        
    
        if self.next_level is not None:
            channels[-1] *= 2 # concatenate skip connection
        self.decoder = nn.Sequential(
            *[ResConvBlock(c_in, c_in, c_out) for c_out, c_in in zip(reversed(channels[:-1]), reversed(channels[1:]))]
        )
    
    def forward(self, x):
        if self.next_level is not None:
            x_enc = self.encoder(x)
            x_next = self.next_level(x_enc)
            x_dec = self.decoder(torch.cat([x_enc, x_next], dim=1))
            return x_dec
        else:
            x_enc = self.encoder(x)
            x_dec = self.decoder(x_enc)
            return x_dec

class UNet(nn.Module):
    def __init__(self,
                *channels: list[int],
                num_blocks: int = 2,
                num_attention_head: int = 4,
                positional_encoding_type = FourierPositionalEncoding2d,
                time_dims=16):
        assert  len(channels) >= 2, 'UNet at least have 2 levels'
        super().__init__()
        unet = UNetLevel(*([channels[-2]] + [channels[-1]] * num_blocks), num_attention_head=num_attention_head, positional_encoding_type=positional_encoding_type, next_level=
                           UNetLevel(*([channels[-1]] + [channels[-1]] * num_blocks), num_attention_head=num_attention_head, positional_encoding_type=positional_encoding_type, next_level=None, time_dims=time_dims if len(channels) == 2 else 0))
        if len(channels) > 2:
            for level in range(len(channels) - 2, 1, -1):
                unet = UNetLevel(*([channels[level-1]] + [channels[level]] * num_blocks), next_level=unet)
            unet = UNetLevel(*([channels[0]] + [channels[1]] * num_blocks), next_level=unet, time_dims=time_dims)
            self.unet = unet
    
    def forward(self, x):
        return self.unet(x)

class CosineSchedule:
    @staticmethod
    def __call__(t):
        return torch.cos(t * torch.pi / 2), torch.sin(t * torch.pi / 2)

class LinearSchedule:
    @staticmethod
    def __call__(t):
        return 1 - t, t

class DDPM(nn.Module):
    def __init__(self, net: nn.Module, timestep_embedding_dim = 16, scheduler=CosineSchedule()):
        super().__init__()

        self.timestep_embed = FourierFeatures(1, timestep_embedding_dim)
        self.net = net
        self.scheduler = scheduler

    def forward(self, x, t, mask = None):
        if mask is None:
            mask = torch.ones(x.shape)
        timestep_embed = self.timestep_embed(t[:, None])[..., None, None].repeat([1, 1, x.shape[2], x.shape[3]])
        zero_time_embed = self.timestep_embed(torch.zeros_like(t)[:, None])[..., None, None].repeat([1, 1, x.shape[2], x.shape[3]])
        timestep_embed = timestep_embed * mask + zero_time_embed * (1-mask)
        return self.net(torch.cat([x, timestep_embed], dim=1))
    
    def loss(self, reals, repeat_factor=1, mask=None):
        """ The costom loss for DDPM model.
        Note1: if it's too slow to load data, repeat them to increase batch size.
        Note2: if you wanna remake some parts of image, use mask to specify the parts.
        Args:
            reals: The clean images.
            repeat_factor: The factor to repeat reals.
            mask: The mask to specify the parts to remake.
        """
        reals = reals.repeat(repeat_factor, 1, 1, 1)
        mask = mask.repeat(repeat_factor, 1, 1, 1) if mask is not None else torch.ones([reals.shape[0], 1, *reals.shape[2:]])
        t = torch.rand(len(reals)).to(self.net.device)
        alphas, sigmas = self.scheduler(t)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        alphas = torch.where(mask==1, alphas, torch.ones_like(alphas))
        sigmas = torch.where(mask==1, sigmas, torch.zeros_like(sigmas))
        noise = torch.randn_like(reals)
        noised_reals = (reals * alphas + noise * sigmas) * mask + reals * (1 - mask)
        targets = noise * alphas - reals * sigmas
        v = self(noised_reals, t, mask)
        return torch.nn.functional.mse_loss(v * mask, targets * mask)

    @torch.no_grad()
    def sample(self, x, steps, eta=1., start=1, mask=None):
        """Draws samples from a model given starting noise.

        Note 1: eta is the amount of noise to add during sampling.
        eta=0, keep old noise and add no new noise (DDIM)
        eta=1, throw away old noise and add new noise (DDPM)

        Note 2: start is the starting time for sampling, should be in [0, 1].
        for example if you wanna increase the quality of an image,
        just add a little noise to it and set start to a small value like 0.1.
        remember when you ganna add a little noise, use a scheduler
        and set start to used time in scheduler.

        Note 3: if you wanna remake some parts of image, use mask to specify the parts.

        Args:
            model: The model to sample from.
            x: The starting noise.
            steps: The number of sampling steps.
            eta: The amount of noise to add during sampling.
            start: The starting time for sampling, should be in [0, 1].
            mask: The mask to specify the parts to remake.
        """
        self.eval()
        ts = x.new_ones([x.shape[0]])

        t = torch.linspace(start, 0, steps + 1)[:-1]
        alphas, sigmas = self.scheduler(t)
        if mask is not None:
            alphas = alphas[:, None, None, None, None].repeat(1, *mask.shape)
            sigmas = sigmas[:, None, None, None, None].repeat(1, *mask.shape)
            alphas = torch.where(mask.unsqueeze(0)==1, alphas, torch.ones_like(alphas))
            sigmas = torch.where(mask.unsqueeze(0)==1, sigmas, torch.zeros_like(sigmas) + 1e-8)

        mask = mask if mask is not None else torch.zeros([x.shape[0], 1, *x.shape[2:]])
        for i in trange(steps):
            v = self(x, ts * t[i], mask).float()
            pred = x * alphas[i] - v * sigmas[i]
            eps = x * sigmas[i] + v * alphas[i]

            # If we are not on the last timestep, compute the noisy image for the
            # next timestep.
            if i < steps - 1:
                # If eta > 0, adjust the scaling factor for the predicted noise
                # downward according to the amount of additional noise to add
                ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                    (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
                adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

                x = (pred * alphas[i + 1] + eps * adjusted_sigma) * mask + x * (1 - mask)

                if eta:
                    x += torch.randn_like(x) * ddim_sigma * mask

        # If we are on the last timestep, output the denoised image
        return pred

class FlowMachine(nn.Module):
    def __init__(self, net: nn.Module, timestep_embedding_dim = 16):
        super().__init__()

        self.timestep_embed = FourierFeatures(1, timestep_embedding_dim)
        self.net = net
        self.scheduler = LinearSchedule()

    def forward(self, x, t, mask = None):
        if mask is None:
            mask = torch.ones(x.shape)
        timestep_embed = self.timestep_embed(t[:, None])[..., None, None].repeat([1, 1, x.shape[2], x.shape[3]])
        zero_time_embed = self.timestep_embed(torch.zeros_like(t)[:, None])[..., None, None].repeat([1, 1, x.shape[2], x.shape[3]])
        timestep_embed = timestep_embed * mask + zero_time_embed * (1-mask)
        return self.net(torch.cat([x, timestep_embed], dim=1))
    
    def loss(self, reals, repeat_factor=1, mask=None):
        """ The costom loss for flow machine model.
        Note1: if it's too slow to load data, repeat them to increase batch size.
        Note2: if you wanna remake some parts of image, use mask to specify the parts.
        Args:
            reals: The clean images.
            repeat_factor: The factor to repeat reals.
            mask: The mask to specify the parts to remake.
        """
        reals = reals.repeat(repeat_factor, 1, 1, 1)
        mask = mask.repeat(repeat_factor, 1, 1, 1) if mask is not None else torch.ones([reals.shape[0], 1, *reals.shape[2:]])
        t = torch.rand(len(reals)).to(self.net.device)
        alphas, sigmas = self.scheduler(t)
        alphas = alphas[:, None, None, None]
        sigmas = sigmas[:, None, None, None]
        alphas = torch.where(mask==1, alphas, torch.ones_like(alphas))
        sigmas = torch.where(mask==1, sigmas, torch.zeros_like(sigmas))
        noise = torch.randn_like(reals)
        noised_reals = (reals * alphas + noise * sigmas) * mask + reals * (1 - mask)
        targets = reals - noise
        v = self(noised_reals, t, mask)
        return torch.nn.functional.mse_loss(v * mask, targets * mask)

    @torch.no_grad()
    def sample(self, x, steps, start=1, mask=None):
        """Draws samples from a model given starting noise.

        Note 1: start is the starting time for sampling, should be in [0, 1].
        for example if you wanna increase the quality of an image,
        just add a little noise to it and set start to a small value like 0.1.
        remember when you ganna add a little noise, use a scheduler
        and set start to used time in scheduler.

        Note 2: if you wanna remake some parts of image, use mask to specify the parts.

        Args:
            model: The model to sample from.
            x: The starting noise.
            steps: The number of sampling steps.
            start: The starting time for sampling, should be in [0, 1].
            mask: The mask to specify the parts to remake.
        """
        self.eval()
        ts = x.new_ones([x.shape[0]])
        mask = mask if mask is not None else torch.zeros([x.shape[0], 1, *x.shape[2:]])

        t = torch.linspace(start, 0, steps + 1)[:-1]

        for i in trange(steps):
            x += self(x, ts * t[i], mask).float() / steps * mask
        return x