from tensorflow.keras import layers, Model
from .sn import SpectralNormalization

def build_discriminator(in_shape=(256,256,3), base=64, dropout=0.3):
    """
    DCGAN Discriminator with SN + BN + LeakyReLU (Fig. 6b).
    Input: real or generated image
    Output: real/fake score (logit)
    """
    img = layers.Input(shape=in_shape, name="disc_input")

    def block(x, f, use_bn=True):
        x = SpectralNormalization(
            layers.Conv2D(f, kernel_size=4, strides=2, padding="same", use_bias=not use_bn)
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x

    x = block(img, base,   use_bn=True)
    x = block(x, base*2,*
