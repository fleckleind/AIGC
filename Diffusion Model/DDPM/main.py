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
# train(ddpm, net, device='cuda', batch_size=16, img_shape=128, n_epochs=100, ckpt_path='/kaggle/working/')


def sample_imgs(ddpm: DDPM, net, output_path, shape, n_sample=81, device='cuda', simple_var=True):
    net = net.to(device)
    net = net.eval()
    with torch.no_grad():
        imgs = ddpm.sample_backward(shape, net, device=device, simple_var=simple_var).detach().cpu()
        imgs = (imgs * 255).clip(0, 255)
        imgs = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c', b1=int(n_sample**0.5))
        imgs = imgs.detach().cpu().numpy().astype(np.uint8)
        imgs = cv2.cvtColor(imgs, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(output_path + 'sample.jpg')
        cv2.imwrite(save_path, imgs)
        
net.load_state_dict(torch.load('/kaggle/input/temp-weight/DDPM-95.pth', weights_only=True))
sample_imgs(ddpm, net, output_path='/kaggle/working/', n_sample=16, shape=(16, 3, 128, 128))
