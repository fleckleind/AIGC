# VQ-VAE
import torch
import torch.nn as nn


class ResidualBlockVAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.conv1(self.relu(x))
        tmp = self.conv2(self.relu(tmp))
        return x + tmp


class VQVAE(nn.Module):
    def __init__(self, input_dim, dim, n_embedding):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlockVAE(dim),  ResidualBlockVAE(dim),
        )
        self.vq_embedding = nn.Embedding(n_embedding, dim)  # add dimension at last
        self.vq_embedding.weight.data.uniform_(-1.0/n_embedding, 1.0/n_embedding)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlockVAE(dim),  ResidualBlockVAE(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
        )
        self.n_downsample = 2

    def forward(self, x):
        # encoder
        ze = self.encoder(x)  # (b,c,h,w)
        embedding = self.vq_embedding.weight.data  # (k,c)
        b, c, h, w = ze.shape
        k, _ = embedding.shape
        # nearest distance: zq <- ze
        embedding_broadcast = embedding.reshape(1, k, c, 1, 1)
        ze_broadcast = ze.reshape(b, 1, c, h, w)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)  # (b,k,h,w)
        nearest_neighbor = torch.argmin(distance, 1)  # (b,h,w)
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)  # (b,h,w,k)->(b,k,h,w)
        # stop back gradient
        decoder_input = ze + (zq - ze).detach()
        # decoder
        x_hat = self.decoder(decoder_input)
        return x_hat, ze, zq

    @torch.no_grad()
    def encode(self, x):
        ze = self.encoder(x)  # (b,c,h,w)
        embedding = self.vq_embedding.weight.data  # (k,c)
        b, c, h, w = ze.shape
        k, _ = embedding.shape
        # nearest distance: zq <- ze
        embedding_broadcast = embedding.reshape(1, k, c, 1, 1)
        ze_broadcast = ze.reshape(b, 1, c, h, w)
        distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)  # (b,k,h,w)
        nearest_neighbor = torch.argmin(distance, 1)  # (b,h,w)
        return nearest_neighbor

    @torch.no_grad()
    def decode(self, discrete_latent):
        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)
        x_hat = self.decoder(z_q)
        return x_hat

    # Shape: [C, H, W]
    def get_latent_HW(self, input_shape):
        C, H, W = input_shape
        return (H // 2**self.n_downsample, W // 2**self.n_downsample)
