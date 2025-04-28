import cv2
import torch
import einops
import numpy as np
import torch.nn as nn


def train(ddpm: DDPM, net, device, batch_size=16, img_shape=128, n_epochs=100, ckpt_path='/kaggle/working/'):
    # training configuration
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(type='CelebAHQ', batch_size=batch_size, img_shape=img_shape)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    tic = time.time()
    for epoch in range(n_epochs):
        total_loss = 0
        for x in dataloader:
            current_batch_size = x.shape[0]
            # sample x_0, t, eps(adding noise)
            x = x.to(device)  # b,c,h,w
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)  # b,
            eps = torch.randn_like(x).to(device)  # Gaussian noise
            x_t = ddpm.sample_forward(x, t, eps)  # x_t: (b,c,h,w) batch for different moment
            # predict noise for different moment
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))  # t.shape->(b,1)
            loss = loss_fn(eps_theta, eps)  # loss between sample and predicted noise
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * current_batch_size
            
        total_loss /= len(dataloader.dataset)
        toc = time.time()
        torch.save(net.state_dict(), os.path.join(ckpt_path, str('DDPM-'+str(epoch)+'.pth')))
        print(f'epoch {epoch} loss: {total_loss} elapsed {(toc - tic):.2f}s')
    print('Done')

ddpm = DDPM(device='cuda', n_steps=1000)
net = DDPMUNet(n_steps=1000, img_shape=(3, 128, 128), pe_dim=128, chans=[16, 32, 64, 128], residual=True)
train(ddpm, net, device='cuda', batch_size=16, img_shape=128, n_epochs=100, ckpt_path='/kaggle/working/')
