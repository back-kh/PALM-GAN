import tensorflow as tf
from tensorflow.keras import layers

def conv3x3(x, filters, name=None):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                      name=None if name is None else f"{name}_conv")(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    x = layers.LeakyReLU(0.2, name=None if name is None else f"{name}_lrelu")(x)
    return x

def residual_block(x, filters, name=None):
    shortcut = x
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                                 name=None if name is None else f"{name}_proj")(shortcut)
    y = conv3x3(x, filters, None if name is None else f"{name}_c1")
    y = conv3x3(y, filters, None if name is None else f"{name}_c2")
    return layers.Add(name=None if name is None else f"{name}_add")([shortcut, y])

class TransformerLayer(layers.Layer):

    def __init__(self, channels, num_heads=4, mlp_ratio=4, drop=0.1, name=None):
        super().__init__(name=name)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=max(1, channels // num_heads),
            attention_axes=(1, 2)
        )
        self.do1 = layers.Dropout(drop)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn1 = layers.Conv2D(channels * mlp_ratio, 1, activation="gelu")
        self.ffn2 = layers.Conv2D(channels, 1)
        self.do2 = layers.Dropout(drop)

    def call(self, x, training=False):
        h = self.attn(self.ln1(x), self.ln1(x), training=training)
        x = x + self.do1(h, training=training)
        h = self.ffn2(self.ffn1(self.ln2(x)))
