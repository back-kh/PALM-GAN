from tensorflow.keras import layers, Model
from .sn import SpectralNormalization

def build_generator(latent_dim=128, out_shape=(256,256,3)):

    H, W, C = out_shape
    z = layers.Input(shape=(latent_dim,), name="latent_input")

    x = SpectralNormalization(layers.Dense(4*4*512, use_bias=False))(z)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((4,4,512))(x)

    for f in [512, 256, 128, 64, 32, 16]:
        x = SpectralNormalization(
            layers.Conv2DTranspose(f, 4, strides=2, padding="same", use_bias=False)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

    out = layers.Conv2DTranspose(C, 4, strides=1, padding="same", activation="tanh")(x)

    return Model(z, out, name="DCGAN_Generator")
