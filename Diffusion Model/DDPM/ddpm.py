import torch

class DDPM():
    # w/o learnable params, no torch.nn.Module
    def __init__(self, device, n_steps: int, min_beta: float = 0.0001, max_beta: float = 0.02):
        # every moment beta/alpha/alpha_bar via n_steps
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  
        alphas, product = 1 - betas, 1
        alpha_bars = torch.empty_like(alphas)
        for i, alpha in enumerate(alphas):
            product *= alpha  # sum of production of alpha
            alpha_bars[i] = product  # save the sum of productions
        self.betas, self.n_steps, self.alphas, self.alpha_bars = betas, n_steps, alphas, alpha_bars

    def sample_forward(self, x, t, eps=None):
        # t: tensor, (batch_size)
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)  # B,1,1,1
        if eps is None:
            eps = torch.randn_like(x)
        # x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon_t
        res = eps * torch.sqrt(1-alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward_step(self, x_t, t, net, simple_var=True):
        n = x_t.shape[0]  # x: b,c,h,w; n=batch_size
        # t_densor: dimension as (b,1) to put into learnable net
        t_tensor = torch.tensor([t]*n, dtype=torch.long).to(device).unsqueeze(1)
        # epsilon: prediction of diffusion model, the only unknown params
        eps = net(x_t, t_tensor)  # predict noise epsilon for mean: b,c,h,w
        if t == 0:
            noise = 0  # t=0 -> z=0
        else:
            if simple_var:
                var = self.betas[t]  # x_0\sim N(0,I)->\sigma_t^2=\beta_t
            else:  # x_0\sim specific distribution
                # \sigma_t^2=(1-\bar{\alpha}_{t-1})/(1-\bar{\alpha}_t)*\beta_t
                var = (1-self.alpha_bars[t-1])/(1-self.alpha_bars[t])*self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)  # \sigma_t z
        # \frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t))
        mean = (x_t-(1-self.alphas[t])/torch.sqrt(1-self.alpha_bars[t])*eps)/torch.sqrt(self.alphas[t])
        x_t = mean + noise
        return x_t

    def sample_backward(self, img_shape, net, device, simple_var=True):
        x = torch.randn(img_shape).to(device)  # x_T: random noise
        net = net.to(device)  # diffusion/denoising model
        for t in range(self.n_steps-1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x
      
