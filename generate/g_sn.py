import tensorflow as tf
from tensorflow.keras import layers

class SpectralNormalization(layers.Wrapper):
    """
    Spectral Normalization wrapper for Conv/Dense layers.
    """
    def __init__(self, layer, power_iterations=1, **kwargs):
        assert hasattr(layer, "kernel"), 
        super().__init__(layer, **kwargs)
        self.power_iterations = power_iterations

    def build(self, input_shape):
        super().build(input_shape)
        self.w = self.layer.kernel
        self.u = self.add_weight(
            shape=(1, self.w.shape[-1]),
            initializer=tf.random_normal_initializer(),
            trainable=False,
            name="sn_u",
        )

    def call(self, inputs, training=None):
        w = tf.reshape(self.w, [-1, self.w.shape[-1]])
        u = self.u
        for _ in range(self.power_iterations):
            v = tf.math.l2_normalize(tf.matmul(u, tf.transpose(w)))
            u = tf.math.l2_normalize(tf.matmul(v, w))
        sigma = tf.matmul(tf.matmul(v, w), tf.transpose(u))
        w_sn = self.w / sigma
        self.layer.kernel.assign(w_sn)
        self.u.assign(u)
        return self.layer(inputs, training=training)
