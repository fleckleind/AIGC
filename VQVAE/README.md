# VQ-VAE
[Neural Discrete Representation Learning](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)  

Vector Quantised-Variational AutoEncoder (VQ-VAE): use vector-quantised (VQ) regularization to solve over-fitting problem in AE. Compared to VAE, VQ-VAE pretains self-supervised encoder-decoder structure, while replaces continuous distribution to discret vectors (attributes) in latent space.

## Embedding Space (Codebook)
Encoder of VQ-VAE compresses the original image $x$ into continuous feature map $z_e$, and converts $z_e$ to discrete vector $z$. For continuous input of decoder, an embedding space is needed to map discrete vector $z$ to continuous embedding $z_q$.  
To simplify vector quantisation, VQ-VAE defines a latent embedding space $e\in R^{K\times D}$, with $K$ as the size of discrete latent space and $D$ as the dimensionality of each latent embedding vector, and posterior categorical distribution $q(z|x)$ defined as one-hot:
```math
q(z=k\vert x)=\left\{\begin{aligned}
1&, k=argmin_j \lVert z_e(x)-e_j \rVert_2\\
0&, otherwise
\end{aligned}\right.
```
VQ-VAE directly convert $z_e$ to $z_q$ via the nearest neighbour method:
```math
z_q(x)=e_k,\quad k=argmin_j\lVert z_e(x)-e_j\rVert_2
```

## Reconstruction Loss
Assuming the embedding space is well-trained, the reconstruction loss of VQ-VAE is defined as follows:
```math
L_{Rec}=\lVert x-D(z_q(x))\rVert_2^2
```
However, this reconstruction loss function cannot propagate the losss to the encoder, because vector quantisation from $z_e$ to $z_q$ has no real gradient. VQ-VAE introduces straight-through estimator as follows:
```math
sg(x)=\left\{\begin{aligned}
x &, forward\\
0 &, backward
\end{aligned}\right.
```
In backward propagation, $sg(z_q(x)-z_e(x))$ term is set as 0, which equals to directly calculate loss based on encoder output $z_e(x)$. And the loss is still based on VQ-embedding $z_q(x)$ in forward propagation.
```math
L_{Rec}=\left\{\begin{aligned}
\lVert x-D(z_q(x))\rVert_2^2 &, forward\\
\lVert x-D(z_e(x))\rVert_2^2 &, backward
\end{aligned}\right.
```
The designed reconstruction loss is then simplified as below:
```math
L_{Rec}=\lVert x-D(z_e(x)+sg(z_q(x)-z_e(x))) \rVert_2^2
```

## Codebook Loss
According to the designed nearest neighbour method, VQ-VAE indirectly updates the codebook/embedding space by optimising the similarity between the encoder output $z_e$ and the decoder input $z_q$ as follows.
```math
L_e = \lVert z_e(x)-z_q(x)\rVert_2^2
```
To differentiate the learning speed of encoder and embedding vector, VQ-VAE use vector quantisation (VQ) in dictionary learning algorithm to optimise embedding space as the first term, with second term as attentive loss to constrain the encoder output $z_e(x)$.
```math
L_e=\lVert sg(z_e(x))-z_q(x) \rVert_2^2+\beta\lVert z_e(x)-sg(z_q(x))\rVert_2^2
```
The total loss is then calculated as follows, with $\beta=4\gamma$:
```math
L=\lVert x-D(z_e(x)+sg(z_q(x)-z_e(x)))\rVert_2^2+\beta\lVert sg(z_e(x))-z_q(x)\rVert_2^2+\gamma\lVert z_e(x)-sg(z_q(x))\rVert_2^2
```
