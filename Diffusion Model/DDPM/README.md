# DDPM
[Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)  

Diffusion Models: replace VAE encoder as fixed noise adding process, and VAE decoder as learnable denoising model, with constant image size. 

## Foward/Diffusion
The input $x_0$ is continuously added Gaussian noise, and $x_T$, the T-process result, is considered as a pure noise image conforming to the standard normal distribution.  
The "Noise Addition" is actually sample image $x_t$ from the normal distribution whose mean is related to the previous image $x_{t-1}$.
```math
x_t \sim \mathcal{N}(\mu_t(x_{t-1}), \sigma_t^2 I)
```
Supposing $x_t \sim \mathcal{N}(\sqrt{1-\beta_t} (x_{t-1}), \beta_t I)$ with $\epsilon_{t-1}\sim\mathcal{N}(0,I)$, the reverse formula is as follows, with the combination formula of $N(0, \sigma_1^2I)$ and $N(0, \sigma_2^2I)$ as $N(0, (\sigma_1^2+\sigma_2^2)I)$:
```math
\begin{align}
x_t&=\sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_{t-1}\\
&=\sqrt{1-\beta_t}(\sqrt{1-\beta_{t-1}}x_{t-2}+\sqrt{\beta_{t-1}}\epsilon_{t-2})+\sqrt{\beta_t}\epsilon_{t-1}\\
&=\sqrt{(1-\beta_t)(1-\beta_{t-1})}x_{t-2}+\sqrt{(1-\beta_t)\beta_{t-1}}\epsilon_{t-2}+\sqrt{\beta_t}\epsilon_{t-1}\\
&=\sqrt{(1-\beta_t)(1-\beta_{t-1})}x_{t-2}+\sqrt{(1-\beta_t)\beta_{t-1}+\beta_t}\epsilon\\
&=\sqrt{(1-\beta_t)(1-\beta_{t-1})}x_{t-2}+\sqrt{1-(1-\beta_t)(1-\beta_{t-1})}\epsilon
\end{align}
```
And the noise adding formula can be summarized as follows, with $\alpha_t=1-\beta_t$ and $\bar{\alpha}_t=\prod_1^t\alpha_i$:
```math
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
```
From $\beta_1=10^{-4}$ to $\beta_T=0.02$, $\beta_i$ is linearly increased, with noise adding formula changing originial images from slow to fast.  

## Reverse/Denoising
A neural network is trained to learn reverse operations and restore $x_T$ back to $x_0$. For generation, we randomly sample noise from the stanard normal distribution, and send it into denoising model.  

As reverse operation of noise adding process cannot directly be found theoretically, the authors use a learnable neural network to fix that. Supposing $x_{t-1}\sim N(\tilde{\mu}_t, \tilde{\beta}_t I)$, the $\tilde{\mu}_t, \tilde{\beta}_t$ are calculated from current moment and image $t, x_t$. With the Bayes formula and summarized noise adding formula, the distribution is calculated as follows:
```math
\begin{align}
q(x_{t-1}|x_t, x_0) &= q(x_t|x_{t-1}, x_0)\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)} \\
\frac{1}{\tilde{\beta}_t \sqrt{2\pi}} exp(-\frac{(x_{t-1}-\tilde{\mu}_t)^2}{2\cdot\tilde{\mu}_t}) &=
\frac{1}{\beta_t \sqrt{2\pi}} exp(-\frac{(x_t-\sqrt{1-\beta_t}x_{t-1})^2}{2\beta_t})\cdot
\frac{1}{(1-\bar{\alpha}_{t-1}) \sqrt{2\pi}} exp(-\frac{(x_t-\sqrt{\bar{\alpha}_{t-1}}x_0)^2}{2(1-\bar{\alpha}_{t-1})})\cdot
(\frac{1}{(1-\bar{\alpha}_t) \sqrt{2\pi}} exp(-\frac{(x_t-\sqrt{\bar{\alpha}_t}x_{t-1})^2}{2(1-\bar{\alpha}_t)}))^{-1}
\end{align}
```
Simplifying the equation, the mean $\tilde{\mu}_t$ and variance $\tilde{\beta}_t$ of the distribution are:
```math
\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t, \quad \tilde{\mu}_t=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_t)
```

## Generation


## Reference
