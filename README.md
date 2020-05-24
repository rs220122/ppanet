# PP-Attention Net: Pyramid Pooling Attention Network.

## Network Architecture

**whole network**<br>
This is the whole network architecture.
 <img src='images/whole_net.png' />

**Pyramid Pooling Attention Module(PPA Module)**<br>
We propose the new module that is called 'Pyramid Pooling Attention (PPA)'.
PPA module include Pyramid Pooling module(PPM) which is proposed in [PSPNet](https://arxiv.org/abs/1612.01105), Atrous Spatial Pyramid Pooling module(ASPP) which is proposed in [Deeplab](https://arxiv.org/abs/1606.00915) and Self-Attention module proposed in [SAGAN](https://arxiv.org/abs/1805.08318).
 <img src='images/ppa.png' />

 **Straight Pyramid Pooling Attention Module(SPPA Module)**<br>
<img src='images/sppa.png' />

## References
1. **tensorflow-DeepLab** <br>
 [link](https://github.com/tensorflow/models/tree/master/research/deeplab)
