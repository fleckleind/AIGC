import cv2
import time
import torch
import einops
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from VQVAE.model import VQVAE
from VQVAE.data import CelebADataset
from GenerationRepo.PerceptualLoss.pcptloss import PerceptualLoss


# dataloader
CELEBA_HQ_DIR = '/kaggle/input/celebahq-resized-256x256/celeba_hq_256'
def get_dataloader(type, batch_size, img_shape=None, dist_train=False,
                   num_workers=4, use_lmdb=False, **kwargs):

    if img_shape is not None:
        kwargs['img_shape'] = img_shape
    dataset = CelebADataset(CELEBA_HQ_DIR, **kwargs)
    if dist_train:  # distributed training, always with data parallel
        sampler = DistributedSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                sampler=sampler, num_workers=num_workers)
        return dataloader, sampler
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)
        return dataloader

# vq-vae training
USE_LMDB = False
def train_vqvae(model: VQVAE, img_shape=None,
                device='cuda', ckpt_path='/kaggle/working/',
                batch_size=64, dataset_type='CelebAHQ',
                lr=1e-3, n_epochs=100, l_w_embedding=1, l_w_commitment=0.25):
    print('batch size:', batch_size)
    dataloader = get_dataloader(dataset_type, batch_size,
                                img_shape=img_shape, use_lmdb=USE_LMDB)
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    mse_loss = nn.MSELoss()
    tic = time.time()
    for e in range(n_epochs):
        total_loss = 0

        for x in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            x_hat, ze, zq = model(x)
            # mse reconstruction loss
            l_reconstruct = mse_loss(x, x_hat)  # Vreconstruction loss
            # perceptual reconstruction loss
            # l_reconstruct = PerceptualLoss(mse_loss, layer_indexs=[14], device=device)(x, x_hat)
            l_embedding = mse_loss(ze.detach(), zq)  # vector quantilization loss
            l_commitment = mse_loss(ze, zq.detach())  # commitment loss
            loss = l_reconstruct + \
                l_w_embedding * l_embedding + l_w_commitment * l_commitment  # alpha=1, beta=0.25
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(model.state_dict(), os.path.join(ckpt_path, str('epoch-'+str(e)+'.pth')))
        print(f'epoch {e} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')

# vq-vae reconstruction and visualization
def reconstruct(model, x, device='cuda', dataset_type='CelebAHQ'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x_hat, _, _ = model(x)
    n = x.shape[0]
    n1 = int(n**0.5)
    x_cat = torch.concat((x, x_hat), 3)
    x_cat = einops.rearrange(x_cat, '(n1 n2) c h w -> (n1 h) (n2 w) c', n1=n1)
    x_cat = (x_cat.clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
    if dataset_type == 'CelebA' or dataset_type == 'CelebAHQ':
        x_cat = cv2.cvtColor(x_cat, cv2.COLOR_RGB2BGR)
    save_path = os.path.join('/kaggle/working/'+dataset_type+'.jpg')
    cv2.imwrite(save_path, x_cat)


img_shape = (3, 128, 128)
vqvae = VQVAE(input_dim=img_shape[0], dim=128, n_embedding=128)
train_vqvae(vqvae, img_shape=(img_shape[1], img_shape[2]), batch_size=64, lr=2e-4, n_epochs=5)
vqvae.load_state_dict(torch.load('/kaggle/input/temp-weight/vq-vae-19.pth', weights_only=True))
dataloader = get_dataloader(type='CelebAHQ', batch_size=16, img_shape=(img_shape[1], img_shape[2]))
reconstruct(vqvae, next(iter(dataloader)).to('cuda'))
