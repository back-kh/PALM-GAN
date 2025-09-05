import tensorflow as tf
from palmgan.models.generator import build_palmgan_generator
from palmgan.models.discriminator import build_palmgan_discriminator

G = build_palmgan_generator((256,256,1))
D = build_palmgan_discriminator((256,256,1))

# Freeze D inside GAN graph
x_in = tf.keras.Input((256,256,1))
y_pred = G(x_in)
D.trainable = False
logits = D([x_in, y_pred])

# composite loss: adversarial + binarization (BCE/L1) + optional SSIM
bce_logits = tf.keras.losses.BinaryCrossentropy(from_logits=True)
bce_pixel  = tf.keras.losses.BinaryCrossentropy()
def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

cgan = tf.keras.Model(x_in, [logits, y_pred])
cgan.compile(
    optimizer=tf.keras.optimizers.Adam(2e-4, 0.5, 0.999),
    loss=[bce_logits, lambda yt, yp: 0.5*bce_pixel(yt, yp) + 0.5*ssim_loss(yt, yp)],
    loss_weights=[1.0, 100.0],
)
