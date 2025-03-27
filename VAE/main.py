import os
import torch
from time import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torch.utils.data.distributed import DistributedSampler
from data import CelebADataset
from model import VAE

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

# VAE loss: MSE and KL
def loss_vae(target, inputs, mean, logvar, kl_weight=0.00025):
    l_rec = F.mse_loss(target, inputs)
    l_kl = torch.mean(-0.5*torch.sum((1+logvar-mean**2-torch.exp(logvar)), dim=1), dim=0)
    loss = l_rec + kl_weight * l_kl
    return loss

# VAE training
def train_vae(model, dataloader, lr, epochs, device, ckpt_path='/kaggle/working/'):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    dataset_length = len(dataloader.dataset)
    begin_time = time()

    for i in range(epochs):
        loss_sum = 0
        for x in dataloader:
            x = x.to(device)
            x_hat, mean, logvar = model(x)
            loss = loss_vae(x, x_hat, mean, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss
        loss_sum /= dataset_length
        training_time = time() - begin_time
        minute, second = int(training_time//60), int(training_time%60)
        print(f'epoch-{i}: loss-{loss_sum}, {minute}:{second}')
        torch.save(model.state_dict(), os.path.join(ckpt_path, str('epoch-'+str(i)+'.pth')))

# VAE reconstuction
def reconstruct_vae(model, device, dataloader, ckpt_path='/kaggle/working/'):
    model.eval()
    batch = next(iter(dataloader))
    x = batch[0:1, ...].to(device)
    output = model(x)[0][0].detach().cpu()
    inputx = batch[0].detach().cpu()
    combined = torch.cat((output, inputx), dim=1)
    img = ToPILImage()(combined)
    img.save(os.path.join(ckpt_path, 'tmp.jpg'))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
dataloader = get_dataloader(type='CelebAHQ', batch_size=16, img_shape=64)
train_vae(model=model, dataloader=dataloader, lr=0.005, epochs=10, device=device)
model.load_state_dict(torch.load('/kaggle/input/temp-weight/vae-9.pth', weights_only=True))
reconstruct_vae(model=model, device=device, dataloader=dataloader)
