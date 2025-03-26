# VAE
[Auto-Encoding Variational Bayes](http://web2.cs.columbia.edu/~blei/fogm/2018F/materials/KingmaWelling2013.pdf)

Variational Auto-Encoder (VAE): use KL regularization to solve over-fitting problem in Auto-Encoder (AE). VAE pretains the encoder-decoder structure and self-training mechanism in AE, while the encoder outputs distribution params ($\mu$ and $\sigma^2$) to replace definte vector $z$, with the constraint/regularization of latent output (similarity between $N(\mu, \sigma^2)$ and $N(0,I)$ ), and the input of the decoder is randomly sampled from $N(\mu, \sigma^2)$. The loss function is shown as below, with first term as reconstruction loss and second term as KL-divergence loss.  
```math
Loss: \lVert x-\hat{x}\rVert^2-KL[N(\mu,\sigma^2)\lVert N(0,I)]
```

## ELBO
VAE: encoder network (inference) as $q_\phi(z|x)$, decoder network (generation) as $p_\theta(z)p_\theta(\hat{x}|z)$, with posterior distribution $p(z|x)\sim N(0,I)$. Based on Bayesian formulation, the KL-divergence:  
```math
\begin{align}
KL[q(z\vert x)\lVert g(z\vert x)] &=\mathbb{E}_{z\sim q}(logq(z\vert x)-log\frac{p(x|z)p(z)}{p(x)})\\
&=\mathbb{E}_{z\sim q}(logq(z\vert x)-logp(x|z)-logp(z)+logp(x))\\
&=\mathbb{E}_{z\sim q}(logq(z\vert x)-logp(x|z)-logp(z))+logp(x)
\end{align}
```
Move $logp(x)$ to the left side:
```math
\begin{align}
logp(x)-KL[q(z\vert x)\lVert g(z\vert x)]&=\mathbb{E}_{z\sim q}(logp(x|z)+logp(z)-logq(z\vert x))\\
logp(x)-KL[q(z\vert x)\lVert g(z\vert x)]&=\mathbb{E}_{z\sim q}(logp(x|z))+\mathbb{E}_{z\sim q}(logp(z)-logq(z\vert x))\\
logp(x)-KL[q(z\vert x)\lVert g(z\vert x)]&=\mathbb{E}_{z\sim q}(logp(x|z))-KL[q(z\vert x)\lVert p(z)]
\end{align}
```
Based on the non-negative property of KL divergence:
```math
\begin{align}
log(x)&\geq\mathbb{E}_{z\sim q}(logp(x|z))-KL[q(z\vert x)\lVert p(z)]\\
ELBO&:=\mathbb{E}_{z\sim q}(logp(x|z))-KL[q(z\vert x)\lVert p(z)]
\end{align}
```
The first term $\mathbb{E}_{z\sim q}(logp(x|z))$ evaluates the quality of image reconstruction, and the second term $KL[q(z\vert x)\lVert p(z)]$ evaluates the similarity between the encoder output $q(z\vert x)$ and posterior distribution $p(z)\sim N(0,I)$ to regularize the latent vector.

## Reparameterization Trick
Sampling $z$ from $Q(z|x)$, as a non-continuous operation withou gradient, VAE cannot back-propagate the error back to encoder. The reparameterization trick is then proposed, with $\epsilon\sim N(0,I)$ multiplied with $\sigma$ then added with $\mu$.

## KL Loss
Considering the multivariate normal distribution with independent components, the univariate normal distribution is derived as follows.
```math
\begin{align}
KL(N(\mu,\sigma^2)\lVert N(0,I))
&=\int\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}(log\frac{e^{-(x-\mu)^2/2\sigma^2}/\sqrt{2\pi\sigma^2}}{e^{-x^2/2}/\sqrt{2\pi}})dx \\
&=\int\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}log\{\frac{1}{\sqrt{\sigma^2}}exp\{\frac{1}{2}[x^2-(x-\mu)^2/\sigma^2]\}\}dx\\
&=\frac{1}{2}\int\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}[-log\sigma^2+x^2-(x-\mu)^2/\sigma^2]dx
\end{align}
```
The first term is the integral of the probability density with coefficience $-log\sigma^2$. The second term is the second order of normal distribution, as $\mu^2+\sigma^2$. According to the definition of variance, the third term is -1.
```math
\begin{align}
\sigma &=\mathbb{E}[(x-\mu)^2]=\int f(x)(x-\mu)^2dx\\
L_{KL} &=\frac{1}{2}(-log\sigma^2+\mu^2+\sigma^2-1)
\end{align}
```

## AE
Auto-Encoder (AE): a self-training mechanism, with encoder compressing image $x$ into a latent vector/embedding $z$, decoder decompressing $z$ into an image $\hat{x}$, and MSE loss function to minimize the difference between $x$ and $\hat{x}$. The decoder is the final generative model, with $z$ sampled from standard normal distribution. However, the encoding $z$ may too short if the sturctures of encoder and decoder are sufficiently complicated, and it will lead the generative model over-fitting.
```math
z = \mathop{argmin}_z\lVert x-\hat{x}\rVert^2
```
