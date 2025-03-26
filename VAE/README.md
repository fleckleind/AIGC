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
The first term $\mathbb{E}_{z\sim q}(log(logp(x|z))$ evaluates the quality of image reconstruction, and the second term $KL[q(z\vert x)\lVert p(z)]$ evaluates the similarity between the encoder output $q(z\vert x)$ and posterior distribution $p(z)\sim N(0,I)$ to regularize the latent vector.

## Reparameterization Trick
Sampling $z$ from $Q(z|x)$, as a non-continuous operation withou gradient, VAE cannot back-propagate the error back to encoder. The reparameterization trick is then proposed, with $\epsilon\sim N(0,I)$ multiplied with $\sigma$ then added with $\mu$.

## AE
Auto-Encoder (AE): a self-training mechanism, with encoder compressing image $x$ into a latent vector/embedding $z$, decoder decompressing $z$ into an image $\hat{x}$, and MSE loss function to minimize the difference between $x$ and $\hat{x}$. The decoder is the final generative model, with $z$ sampled from standard normal distribution. However, the encoding $z$ may too short if the sturctures of encoder and decoder are sufficiently complicated, and it will lead the generative model over-fitting.
```math
z = \mathop{argmin}_z\lVert\hat{x}-x\rVert^2
```
