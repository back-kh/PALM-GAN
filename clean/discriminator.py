from tensorflow.keras import layers, Model

def disc_block(x, filters, use_bn=True, drop=False, name=None):

    x = layers.Conv2D(filters, 3, strides=2, padding="same", use_bias=not use_bn,
                      name=None if name is None else f"{name}_conv")(x)
    if use_bn:
        x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = layers.LeakyReLU(0.2, name=None if name is None else f"{name}_lrelu")(x)
    if drop:
        x = layers.Dropout(0.25, name=None if name is None else f"{name}_drop")(x)
    return x

def build_palmgan_discriminator(input_shape=(256, 256, 1)):
    x_cond = layers.Input(shape=input_shape, name="x_cond")   # degraded input
    y_in   = layers.Input(shape=input_shape, name="y_in")     # real GT or G(x)

    z = layers.Concatenate(name="xy_concat")([x_cond, y_in])

    z = disc_block(z,  64, use_bn=False, drop=True,  name="d64")   # first: no BN
    z = disc_block(z, 128, use_bn=True,  drop=False, name="d128")
    z = disc_block(z, 256, use_bn=False, drop=True,  name="d256")
    # stride 1 for deeper receptive field without over-downsampling
    z = layers.Conv2D(512, 3, strides=1, padding="same", use_bias=False, name="d512_conv")(z)
    z = layers.BatchNormalization(name="d512_bn")(z)
    z = layers.LeakyReLU(0.2, name="d512_lrelu")(z)
    z = layers.Dropout(0.25, name="d512_drop")(z)

    logits = layers.Conv2D(1, 3, strides=1, padding="same", name="logits")(z)
    return Model([x_cond, y_in], logits, name="PALM_GAN_D")
