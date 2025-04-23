# DDPM
[Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)  

Diffusion Models: replace VAE encoder as fixed noise addition, and VAE decoder as learnable denoising model, with constant image size. 

## Foward
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

## Reverse
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
Simplifying coefficience and $x_{t-1}$ terms of the equation, the mean $\tilde{\mu}_t$ and variance $\tilde{\beta}_t$ of the distribution are:
```math
\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t, \quad \tilde{\mu}_t=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_t)
```  
With the only unknown params $\epsilon_t$ in $\tilde{\mu}_t$ distribution, the ambition of denoising model can be simplified as follows:
```math
L=\lVert \epsilon_t-\epsilon_{\theta}(x_t, t)\rVert^2
```

### Supplement
The diffusion model $p_\theta(x_0)$, with trainable parameters $\theta$, describes the probability of generating data $x_0$ in the reverse process.
```math
p_\theta(x_{0:T})=\int p_\theta(x_{0:T})dx_{1:T}=p(x_T)\prod_{t-1}^T p_\theta(x_{t-1}|x_t)
```
And the recursion formula is shown as follows, with $\Sigma_\theta(x_t, t)=\tilde{\beta}_t I$
```math
p_\theta(x_{t-1}|x_t)=N(x_{t-1};\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
```
The original ambition of diffusion model is to maximise the probability $p_\theta(x_0)$, which is equal to minimise $-log p_\theta(x_0)$ and the ambition is described as follows via variational lower bound in VAE:
```math
L_{VLB}=E[D_{KL}(q(x_T|x_0)\lVert p_\theta(x_T))+
\sum_{t=2}^T D_{KL}(q(x_{t-1}|x_t, x_0)\lVert p_\theta(x_{t-1}|x_t))-
log p_\theta(x_0|x_1)]
```
while $D_{KL}$ as KL divergence between distribution $p$ and $q$, $q(x_{1:T}|x_0)$ as forward noise adding process, $q(x_{t-1}|x_t, x_0)$ as theoretical denoising process, and $p_\theta(x_{t-1}|x_t)$ as reverse process. The first term can be ignored for no learnable params, the second term maximize the similarity between denoising process and reverse noise adding process, and the third term is to restore $x_0$ with known $x_1$.  
As the computation of KL Divergen defined as follows,
```math
D_{KL}(P\lVert Q)=log\frac{\sigma_2}{\sigma_1}+\frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2}-\frac{1}{2}
```
For the second term $D_{KL}(q(x_{t-1}|x_t, x_0)\lVert p_\theta(x_{t-1}|x_t))$, the KL divergence is simplified as follows without considering variance term,
```math
D_{KL}(q(x_{t-1}|x_t, x_0)\lVert p_\theta(x_{t-1}|x_t)) \rightarrow
\frac{1}{2\tilde{\beta}_t^2}\lVert \mu_\theta(x_t,t)-\tilde{\mu}_t(x_t,t)\rVert^2
```
Considering the mean formula, the optimized ambition can be further simplified as follows:
```math
D_{KL}(q(x_{t-1}|x_t, x_0)\lVert p_\theta(x_{t-1}|x_t))\rightarrow
\lVert \epsilon_t-\epsilon_\theta(x_t,t)\rVert^2
```
As for the third term:
```math
-log p_\theta(x_0|x_1)]=-log\frac{1}{\sqrt{2\pi}\tilde{\beta}_1^2}+
\frac{\lVert x_0-\mu_theta(x_!,1)\rVert^2}{2\tilde{\beta}_1^2}
```
the learnable latter term can be simplified as follows:
```math
\begin{align}
\frac{\lVert x_0-\mu_theta(x_1,1)\rVert^2}{2\tilde{\beta}_1^2}
&\rightarrow \lVert x_0-\frac{1}{\sqrt{\alpha_1}}(x_1-\frac{1-\alpha_1}{\sqrt{1-\bar{\alpha}_1}}\epsilon_\theta(x_1,1))\rVert^2\\
&\rightarrow \lVert x_0-\frac{1}{\sqrt{\alpha_1}}(\sqrt{\bar{\alpha}_1}x_0+\sqrt{1-\bar{\alpha}_1}\epsilon_1-\frac{1-\alpha_1}{\sqrt{1-\bar{\alpha}_1}}\epsilon_\theta(x_1,1))\rVert^2\\
&\rightarrow \lVert \sqrt{1-\bar{\alpha}_1}\epsilon_1-\frac{1-\alpha_1}{\sqrt{1-\bar{\alpha}_1}}\epsilon_\theta(x_1,1))\rVert^2\\
&\rightarrow \lVert \epsilon_1-\epsilon_\theta(x_1,1)\rVert^2
\end{align}
```

## Training and Sampling
Randomly sample a data $x_0$ and moment $t$ from training set and $Uniform(\{1,\ldots,T\})$, with moment chosen to simplify the training process. Then randomly generate noise $\epsilon\sim N(0, I)$ for forward process and get $x_t$, and take gradient descent step on following loss:
```math
\nabla_{\theta}\lVert\epsilon-\epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\rVert^2
```
The reverse process denoise any noisy image to generate image. Firstly sample $x_T\sim N(0,I)$, and calculate the mean and variance for moment from $T$ to 1, with $z\sim N(0,I)$:
```math
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t))+\sigma_t z
```

## Reference
[What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#connection-with-stochastic-gradient-langevin-dynamics)  
[Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/pdf/2208.11970)  
