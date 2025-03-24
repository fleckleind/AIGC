# Perceptual Loss
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155)  
Perceptual loss functions are based on high-level feature extracted from a loss network $\phi$ pretrained for image classification (16-layer VGG), measuring high-level perceptual and semantic differences between images.

## Feature Reconstruction Loss
The feature reconstruction loss is the (squared, normalized) Euclidean distance between feature representations computed by loss network $\phi$. 
$\phi_j(x)$ represents as the activation of the $jth$ layer of the loss network when processing image $x$, with the shape of feature map as $C_j\times H_j\times W_j$.  
```math
l_{feat}^{\phi, j}(\hat{y}, y)=\frac{1}{C_jH_jW_j}\lVert\phi_j(\hat{y})-\phi_j(y)\rVert_2^2
```
## Style Reconstruction Loss
Gram matrix $G_j^\phi(x)$ is defined to be the $C_j\times C_j$ matrix, capturing the (covariance) relationship between different channel in $jth$ feature map to reflect information of context and style.  
```math
G_j^\phi(x)_{c,c'}=\frac{1}{C_jH_jW_j}\sum_{h=1}^{H_j}\sum_{w=1}^{W_j}\phi_j(x)_{h,w,c}\phi_j(x)_{h,w,c'}
```

Based on the Gram matrix, style reconstruction loss is defined as follows, with $\lVert\cdot\rVert_F$ as Frobenius norm (computing the square root of the sum of the element-wise squared differences of two matrices).  
```math
l_{style}^{\phi, j}(\hat{y}, y)=\lVert G_j^phi(\hat{y})-G_j^\phi(y)\rVert_F^2
```
