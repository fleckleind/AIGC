# VAE
import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim=128, chans=[16, 32, 64, 128, 256]):
        super(VAE, self).__init__()
        # encoder
        self.latent_dim = latent_dim  # multi-dimensional normal distribution
        modules, in_chan, img_size = [], 3, 64
        for out_chan in chans:
            modules.append(nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_chan),
                nn.ReLU(inplace=True),
            ))
            in_chan, img_size = out_chan, img_size//2
        self.encoder = nn.Sequential(*modules)
        # encoder output normal distribution params
        self.mean_linear = nn.Linear(in_chan*img_size*img_size, latent_dim)
        self.var_linear = nn.Linear(in_chan*img_size*img_size, latent_dim)
        
        # decoder
        modules = []
        self.dec_projection = nn.Linear(latent_dim, in_chan*img_size*img_size)
        self.dec_input_size = (in_chan, img_size, img_size)
        for i in range(len(chans)-1, 0, -1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(chans[i], chans[i-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(chans[i-1]),
                nn.ReLU(inplace=True),
            ))
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(chans[0], chans[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(chans[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(chans[0], 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ))
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        encoded = torch.flatten(self.encoder(x), 1)  # (b,c,h,w)->(b,c*h*w)
        mean, logvar = self.mean_linear(encoded), self.var_linear(encoded)  # (b,c*h*w)->(b,latent_dim)
        # eps: randomly sample, std: E_{z~q}(log p(x|z))
        eps, std = torch.rand_like(logvar), torch.exp(logvar / 2)  # reparameterized trick
        z = self.dec_projection(eps * std + mean).reshape(-1, *self.dec_input_size)
        out = self.decoder(z)
        return out, mean, logvar

    def sample(self, device='cuda'):
        # generate image from randomly sampling z~N(0,I)
        z = torch.randn(1, self.latent_dim).to(device)
        z = self.decoder_projection(z).reshape(-1, *self.dec_input_size)
        out = self.decoder(x)
        return out
