# Official Implementation
Repository: [taming-transformers](https://github.com/CompVis/taming-transformers/), [VQGAN-pytorch](https://github.com/dome272/VQGAN-pytorch)  

## Encoder and Decoder
U-Net block consisting of residual connection block and attnetion module, with downsampling as convolution and upsampling as nearest interpolation.
```math
Attn(x)=V(Softmax(\frac{QK}{\sqrt{c}}))^T+x,\quad QK=\sum_c q[b,i,c]k[b,c,j]
```

## Codebook
Intial an embedding layer, calculate the distance between encoder output $z$ and all embedding $d$, use $argmin$ to get index of the nearest embedding, with $.detach()$ as stop-gradient operation.  

## Perceptual Loss
LPIPS (learned perceptual image patch similarity): normalize input and target, send into VGG16 and get feature maps in layer [3, 8, 15, 22, 29]. After use MSE to calculate the difference between the feature maps from input and target, LPIPS normalize and average the results, then add results from all layers.  

## GAN Loss
Referring [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/), the descriminator sequentially downsamples reconstructed image with convolution (4 times), with channel size up to $64\times8$, then resizes the channel dimension back to 1.

## Conditioned Transformer
Compress input $x$ and constraint $c$ to compressed embeddings via different VQ-GAN, randomly substitue constraint embedding like Dropout, then concatenate the processed embeddings and use GPT2 to obtain reconstruted patches.  
GPT2, Transformer only with decoder, uses sequential token embedding layer and several Transformer blocks to process input.

## Sliding Window Manner
Model firstly processes conditional (semantic segmentation) inputs via VQ-GAN and get $c_{indices}$. And $z_{indices}$ is randomly generated to initialize compressed embedding.  
Suppose that the number of $16\times16$ windows is 4, the relative positions (range) of column and row index are seperated as 3 situations (start=0/size-16/index-8).  
Reshape the partial embeddings to unidimensional and concatenate, GPT2 is used to generate compressed patch, and push the probability bcak based on relative position.
At the end of the sliding window processionm, the pixel sampled from compressed patch is then push into $z_{indices}$ to realise synthetic step.


