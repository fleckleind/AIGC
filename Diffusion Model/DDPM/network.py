import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    # max_seq_len: n_steps, d_model: hidden layer dimension
    def __init__(self, max_seq_len: int, d_model: int):
        super(PositionalEncoding, self).__init__()
        # sin/cos positional encoding array
        assert d_model % 2 == 0  
        pe = torch.zeros(max_seq_len, d_model)
        # positional sequence
        i_seq = torch.linspace(0, max_seq_len-1, max_seq_len)  # max_seq_len feature dict 
        j_seq = torch.linspace(0, d_model-2, d_model//2)  # sin/cos for model dimension
        # grid coordinance: relative position
        pos, i_2 = torch.meshgrid(i_seq, j_seq)  # setting array for sin/cos array
        pe_2i = torch.sin(pos/10000**(i_2/d_model))  # sin positional definition
        pe_2i_1 = torch.cos(pos/10000**(i_2/d_model))  # cos positional definition
        # concatenate in d_model wise, and reshape to original size
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)
        # initial data with pe, and w/o training
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)


class ResidualDDPM(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super(ResidualDDPM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),)
        self.resd = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.resd(x) + self.conv(x)


class ConvNet(nn.Module):
    # noise prediction via noisy iamge x and time_step t
    # diffusion steps, image chan, hidden layer(x/t) chan
    def __init__(self, n_steps, in_ch, pe_dim=10, hid_chans=[10, 20, 40], insert_t_to_all_layers=False):
        super(ConvNet, self).__init__()
        self.pe = PositionalEncoding(n_steps, pe_dim)  # positional encoding
        self.pe_linears = nn.ModuleList()
        # insert positional encoding "t" to all layers
        self.all_t = insert_t_to_all_layers
        if not self.all_t:  # only input layer
            # adjust positional encoding with x.chan
            self.pe_linears.append(nn.Linear(pe_dim, in_ch))
        pre_ch, self.residual_blocks = in_ch, nn.ModuleList()
        for chan in hid_chans:
            self.residual_blocks.append(ResidualDDPM(pre_ch, chan))
            if self.all_t:
                self.pe_linears.append(nn.Linear(pe_dim, pre_ch))
            else:
                self.pe_linears.append(None)
            pre_ch = chan
        self.output_layer = nn.Conv2d(pre_ch, in_ch, 1)

    def forward(self, x, t):
        # x: b,c,h,w; t: b
        n = t.shape[0]  # batch_size
        t = self.pe(t)  # b, pe_dim
        for m_x, m_t in zip(self.residual_blocks, self.pe_linears):
            if m_t is not None:
                # b,pe_dim->b,c->b,c,1,1
                pe = m_t(t).reshape(n, -1, 1, 1)  
                x = x + pe
            x = m_x(x)
        x = self.output_layer(x)
        return x


class DDPMUNetBlock(nn.Module):
    def __init__(self, shape, in_ch, out_ch, residual=False):
        super(DDPMUNetBlock, self).__init__()
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Sequential(
            nn.LayerNorm(shape),  # normalization: c,h,w
            nn.Conv2d(in_ch, out_ch, 3, 1, 1), self.act_fn, 
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),)
        self.residual = residual
        if self.residual:
            self.resd = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            out += self.resd(x)
        out = self.act_fn(out)
        return out


class DDPMUNet(nn.Module):
    def __init__(self, n_steps, img_shape, pe_dim=16, chans=[16, 32, 64, 128], residual=False):
        super(DDPMUNet, self).__init__()
        self.pe = PositionalEncoding(n_steps, pe_dim)
        # network architecture
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.encoders, self.pe_linears_enc = nn.ModuleList(), nn.ModuleList()
        self.decoders, self.pe_linears_dec = nn.ModuleList(), nn.ModuleList()
        # encoder structure
        pre_ch, h, w = img_shape[0], img_shape[1], img_shape[2]
        for chan in chans[0:-1]:
            self.pe_linears_enc.append(nn.Sequential(nn.Linear(pe_dim, pre_ch)))
            self.encoders.append(
                nn.Sequential(DDPMUNetBlock((h, w), pre_ch, chan, residual=residual),
                              DDPMUNetBlock((h, w), chan, chan, residual=residual),))
            self.downs.append(nn.Conv2d(chan, chan, 3, 2, 1))
            pre_ch, h, w = chan, h // 2, w // 2
        # bottleneck
        self.pe_linears_mid = nn.Linear(pe_dim, pre_ch)
        self.bottleneck = nn.Sequential(
            DDPMUNetBlock((h, w), pre_ch, chans[-1], residual=residual),
            DDPMUNetBlock((h, w), chans[-1], chans[-1], residual=residual),)
        # decoder structure
        pre_ch, h, w = chans[-1], h * 2, w * 2
        for chan in chans[-2::-1]:
            self.pe_linears_dec.append(nn.Linear(pe_dim, pre_ch))
            self.ups.append(nn.ConvTranspose2d(pre_ch, chan, 4, 2, 1))
            # skip connection, with 2x channel size
            self.decoders.append(
                nn.Sequential(DDPMUNetBlock((h, w), pre_ch, chan, residual=residual),
                              DDPMUNetBlock((h, w), chan, chan, residual=residual),))
            pre_ch, h, w = chan, h * 2, w * 2
        self.output = nn.Conv2d(pre_ch, img_shape[0], 1)
    
    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        # encoder and corresponding features
        encoder_feats = []
        for pe_linear, encoder, down in zip(self.pe_linears_enc, self.encoders, self.downs):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = encoder(x + pe)
            encoder_feats.append(x)
            x = down(x)
        # bottleneck
        pe = self.pe_linears_mid(t).reshape(n, -1, 1, 1)
        x = self.bottleneck(x + pe)
        # decoder with skip connection
        for pe_linear, decoder, up, encoder_feat in zip(self.pe_linears_dec, self.decoders, self.ups, encoder_feats[::-1]):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = up(x)
            x = torch.cat([encoder_feat, x], dim=1)
            x = decoder(x + pe)
        x = self.output(x)
        return x
      
