import tensorflow as tf
from tensorflow.keras import layers, Model

# ---------- Core blocks ----------
def conv3x3(x, filters, name=None):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=None if name is None else f"{name}_conv")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = layers.LeakyReLU(0.2, name=None if name is None else f"{name}_lrelu")(x)
    return x

def residual_conv_block(x, filters, name=None):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                                 name=None if name is None else f"{name}_proj")(shortcut)
    y = conv3x3(x, filters, name=None if name is None else f"{name}_c1")
    y = conv3x3(y, filters, name=None if name is None else f"{name}_c2")
    out = layers.Add(name=None if name is None else f"{name}_add")([shortcut, y])
    return out

class TransformerBlock(layers.Layer):

    def __init__(self, channels, num_heads=4, mlp_ratio=4, drop_rate=0.0, name=None):
        super().__init__(name=name)
        self.norm1  = layers.LayerNormalization(epsilon=1e-6)
        self.attn   = layers.MultiHeadAttention(num_heads=num_heads, key_dim=channels//num_heads,
                                                attention_axes=(1, 2))
        self.drop1  = layers.Dropout(drop_rate)
        self.norm2  = layers.LayerNormalization(epsilon=1e-6)
        self.mlp1   = layers.Conv2D(channels*mlp_ratio, 1, activation="gelu")
        self.mlp2   = layers.Conv2D(channels, 1)
        self.drop2  = layers.Dropout(drop_rate)

    def call(self, x, training=False):
        h = self.norm1(x)
        h = self.attn(h, h, training=training)
        x = x + self.drop1(h, training=training)

        h = self.norm2(x)
        h = self.mlp2(self.mlp1(h))
        x = x + self.drop2(h, training=training)
        return x

def attention_block(x, filters, name=None):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                                 name=None if name is None else f"{name}_proj")(shortcut)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name=None if name is None else f"{name}_c1")(x)
    y = layers.BatchNormalization(name=None if name is None else f"{name}_bn1")(y)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name=None if name is None else f"{name}_c2")(y)
    y = layers.BatchNormalization(name=None if name is None else f"{name}_bn2")(y)
    y = layers.LeakyReLU(0.2, name=None if name is None else f"{name}_lrelu")(y)

    y = TransformerBlock(filters, num_heads=max(1, filters // 64), mlp_ratio=4,
                         drop_rate=0.1, name=None if name is None else f"{name}_trans")(y)

    y = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name=None if name is None else f"{name}_c3")(y)
    y = layers.BatchNormalization(name=None if name is None else f"{name}_bn3")(y)

    out = layers.Add(name=None if name is None else f"{name}_add")([shortcut, y])
    return out

def build_palmgan_generator(input_shape=(256, 256, 1)):
    inp = layers.Input(shape=input_shape, name="degraded_input")

    # Downsampling path (Conv -> Residual + MaxPool), with periodic attention blocks
    d1 = residual_conv_block(conv3x3(inp,   64, "d1_in"),  64,  "d1")         # 256 -> 256
    p1 = layers.MaxPooling2D()(d1)                                             # 256 -> 128

    d2 = residual_conv_block(conv3x3(p1,   128, "d2_in"), 128,  "d2")         # 128 -> 128
    p2 = layers.MaxPooling2D()(d2)                                             # 128 -> 64

    d3 = residual_conv_block(conv3x3(p2,   256, "d3_in"), 256,  "d3")         # 64 -> 64
    p3 = layers.MaxPooling2D()(d3)                                             # 64 -> 32

    # Attention in encoder (as in the figure)
    d4_pre = conv3x3(p3, 512, "d4_in")
    d4 = attention_block(d4_pre, 512, "d4_att")                                # 32 -> 32
    p4 = layers.MaxPooling2D()(d4)                                             # 32 -> 16

    # Bottleneck with attention
    bn_pre = conv3x3(p4, 1024, "bn_in")
    bn = attention_block(bn_pre, 1024, "bn_att")                               # 16 -> 16

    # Upsampling path (ConvT -> concat skip -> convs), with an attention block near the top
    u4 = layers.Conv2DTranspose(512, 3, strides=2, padding="same", use_bias=False)(bn)  # 16->32
    u4 = layers.Concatenate()([u4, d4])                                        # skip
    u4 = residual_conv_block(conv3x3(u4, 512, "u4_in"), 512, "u4")

    u3 = layers.Conv2DTranspose(256, 3, strides=2, padding="same", use_bias=False)(u4)  # 32->64
    u3 = layers.Concatenate()([u3, d3])
    u3 = attention_block(conv3x3(u3, 256, "u3_in"), 256, "u3_att")

    u2 = layers.Conv2DTranspose(128, 3, strides=2, padding="same", use_bias=False)(u3)  # 64->128
    u2 = layers.Concatenate()([u2, d2])
    u2 = residual_conv_block(conv3x3(u2, 128, "u2_in"), 128, "u2")

    u1 = layers.Conv2DTranspose(64, 3, strides=2, padding="same", use_bias=False)(u2)   # 128->256
    u1 = layers.Concatenate()([u1, d1])
    u1 = attention_block(conv3x3(u1, 64, "u1_in"), 64, "u1_att")

    # Output (binarization map)
    out = layers.Conv2D(1, 1, padding="same", activation="sigmoid", name="bin_out")(u1)

    return Model(inp, out, name="PALMGAN_G")
