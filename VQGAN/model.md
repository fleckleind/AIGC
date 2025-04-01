# Official Implementation
Official Implementation Repository: [taming-transformers](https://github.com/CompVis/taming-transformers/)  
Encoder and Decoder: U-Net block consisting of residual connection block and attnetion module, with downsampling as convolution and upsampling as nearest interpolation.
```math
Attn(x)=V(Softmax(\frac{QK}{\sqrt{c}}))^T+x,\quad QK=\sum_c q[b,i,c]k[b,c,j]
```
Codebook: intial an embedding layer, calculate the distance between encoder output $z$ and all embedding $d$, use $argmin$ to get index of the nearest embedding, with $.detach()$ as stop-gradient operation.  
LPIPS (learned perceptual image patch similarity): normalize input and target, send into VGG16 and get feature maps in layer [3, 8, 15, 22, 29]. After use MSE to calculate the difference between the feature maps from input and target, LPIPS normalize and average the results, then add results from all layers.  
