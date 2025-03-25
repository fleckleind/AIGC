# VAE
[Auto-Encoding Variational Bayes](http://web2.cs.columbia.edu/~blei/fogm/2018F/materials/KingmaWelling2013.pdf)  
[Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908)  
AutoEncoder (AE): a self-training mechanism, with encoder compressing image $x$ into a latent vector/embedding $z$, decoder decompressing $z$ into an image $\hat{x}$, and MSE loss function to minimize the difference between $x$ and $\hat{x}$. The decoder is the final generative model, with $z$ sampled from standard normal distribution.
```\math
z = argmin_z\lVert\hat{x}-x\rVert^2
```
