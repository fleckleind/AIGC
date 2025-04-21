# DDPM
[Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)  

Diffusion Model: replace VAE encoder as noise adding process, and VAE decoder as denoising process, with fixed-resolution image.

## Forward/Diffusion Process
The input $x_0$ is gradually added Gaussian noise and becomes the image fixing  standard normalized distrubution, according to a variance schedule $\beta_1, \ldots, \beta_T$ as follows:
```math
q(x_{1:T}|x_0):=\prod_{t=1}^T q(x_t|x_{t-1}), \quad q(x_t|x_{t-1}):=N(x_t;\sqrt{1-\beta_t} x_{t-1}, \beta_t I)
```
Given $x_0$, which is sampled from training set, the noisy image $x_t$\sim $t-process$ is defined as follows, with $\epsilon~N(0,I)$.
```math
\begin{align}
x_t & =\sqrt{1-\beta_t} x_{t-1}+\sqrt{\beta_t} \epsilon_{t-1} \\
& =\sqrt{1-\beta_t} (\sqrt{1-\beta_{t-1}}x_{t-2}+\sqrt{\beta_{t-1}} \epsilon_{t-2})+\sqrt{\beta_t} \epsilon_{t-1} \\
& =\sqrt{(1-\beta_t)(1-\beta_{t-1})} x_{t-2}+\sqrt{(1-\beta_t)\beta_{t-1}} \epsilon_{t-2}+\sqrt{\beta_t} \epsilon_{t-1} \\
& =\sqrt{(1-\beta_t)(1-\beta_{t-1})} x_{t-2}+\sqrt{(1-\beta_t)\beta_{t-1}+\beta_t} \epsilon \\
& =\sqrt{(1-\beta_t)(1-\beta_{t-1})} x_{t-2}+\sqrt{1-(1-\beta_t)(1-\beta_{t-1})} \epsilon
\end{align}
```
And the formulla is finally as follows, with $\alpha_t=1-\beta_t$ and ${\bar{\alpha}}_t=\prod_i^t\alpha_i$, with raw image changing speed from slow to fast:
```math
x_t=\sqrt{{\bar{\alpha}}_t} x_0+\sqrt{1-{\bar{\alpha}}_t} \epsilon
```

## Reverse/Denoising Process
In the reverse process, a neural network is supposed to train as a model denoising $T$ adding noise, getting $x_0$ from $x_T$:
```math
p_{\theta}(x_{0:T}):=p(x_T)\prod_{t=1}^T p_{\theta}(x_{t-1}|x_t), \quad p_{\theta}(x_{t-1}|x_t):=N(x_{t-1}; \mu_{\theta}(x_t, t), \Sigma_{\theta}(x_t,t))
```
As $x_{t-1}\sim N({\tilde{\mu}}_t, {\tilde{\beta}}_t I), and ${\tilde{\mu}}_t, {\tilde{\beta}}_t$ fitted by $t$ and $x_t$, the neural network needs $t, x_t$ as inputs to describe denoising process. Based on Bayes formula, 
```math
q(x_{t-1}|x_t, x_0)=q(x_t|x_{t-1}, x_0)\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}
```
where $q(x_t|x_{t-1}, x_0)=N(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$





## Reference
[Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/pdf/2208.11970)
