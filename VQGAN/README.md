# VQ-GAN
[Taming Transformers for High-Resolution Image Synthesis](http://openaccess.thecvf.com/content/CVPR2021/papers/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.pdf)  

Based on VQ-VAE, VQ-GAN uses perceptual loss to substitute original MSE for reconstruction loss, and introduces patch-based GAN loss. The total loss is defined as follows:
```math
Q^*=\mathop{argmin}_{E,G,Z}\,\mathop{max}_{D}\,E_{x\sim p(x)}(L_{VQ}(E,G,\mathop{Z})+\lambda L_{GAN}(\{E,G,\mathcal{Z}\},D))
```
The adaptive weight $\lambda$ is computed as follows, with $\delta=10^{-6}$:
```math
\lambda=\frac{\nabla_{G_L}[L_{Rec}]}{\nabla_{G_L}[L_{GAN}]+\delta}
```

## VQ Loss
Based on [VQ-VAE](https://github.com/fleckleind/GenerationRepo/tree/main/VQVAE),
the VQ loss in VQ-GAN contains reconstruction loss, codebook loss, and attentive (commitment) loss:
```math
\begin{align}L_{VQ}(E,G,Z)
&=\lVert x-D(z_e(x)+sg(z_q(x)-z_e(x)))\rVert\\
&+\lVert sg(z_e(x))-z_q(x)\rVert + \beta\lVert sg(z_q(x)-z_e(x)\rVert
\end{align}
```
To enhance spatial representation of VQ-GAN, the receptual loss is introduced to replace original MSE for reconstruction penalty in VQ-VAE. The brief introduction of receptual loss can be found [here](https://github.com/fleckleind/GenerationRepo/tree/main/PerceptualLoss).

## GAN Loss
VQ-GAN introduces an adversarial training procedure with a patch-based discriminator $D$ to differentiate real and reconstructed images:
```math
L_{GAN}(\{E,G,Z\},D)=log D(x) + log(1-D(\hat{x}))
```

## Latent Transformer and Conditioned Synthesis
To better use discrete vectors in codebook, VQ-GAN replace the PixelCNN in VQ-VAE by Transformer, which is better to process sequential or unidimensional information than CNN. Given indices $s_{\textless i}$, the learnable Transformer predicts distribution of possible next indices and the likelihood of full representation.
```math
p(s)=\prod_i p(s_i|s_{\textless i}),
\quad L_{Transformer}=E_{x\sim p(X)}[-log p(s)]
```
To control over the generation process with provided additional information $c$, like single label or another image, VQ-GAN referring conditional Transformer (decoder-only strategy), sets $c$ as class token and learns the likelihood of the sequence:
```math
p(s|c)=\prod_i p(s_i|s_{\textless i}, c)
```
It notes that if the conditioning information has spatial extent, VQ-GAN first learns another VQ-GAN to obtain aother codebook $Z_c$, with representation $r\in\\{0,\ldots,|Z_c|-1\\}^{h_c\times w_c}$.
