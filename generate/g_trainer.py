import tensorflow as tf
from models.generator import build_generator
from models.discriminator import build_discriminator

class DCGANTrainer:
    def __init__(self, latent_dim=128, img_shape=(256,256,3), lr=2e-4):
        self.latent_dim = latent_dim
        self.G = build_generator(latent_dim, img_shape)
        self.D = build_discriminator(img_shape)

        self.opt_g = tf.keras.optimizers.Adam(lr, 0.5, 0.999)
        self.opt_d = tf.keras.optimizers.Adam(lr, 0.5, 0.999)

        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(self, real_imgs):
        batch_size = tf.shape(real_imgs)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        # --- Train Discriminator ---
        with tf.GradientTape() as td:
            fake_imgs = self.G(noise, training=True)
            d_real = self.D(real_imgs, training=True)
            d_fake = self.D(fake_imgs, training=True)
            d_loss_real = self.loss_fn(tf.ones_like(d_real), d_real)
            d_loss_fake = self.loss_fn(tf.zeros_like(d_fake), d_fake)
            d_loss = d_loss_real + d_loss_fake
        grads_d = td.gradient(d_loss, self.D.trainable_variables)
        self.opt_d.apply_gradients(zip(grads_d, self.D.trainable_variables))

        # --- Train Generator ---
        with tf.GradientTape() as tg:
            fake_imgs = self.G(noise, training=True)
            d_fake = self.D(fake_imgs, training=True)
            g_loss = self.loss_fn(tf.ones_like(d_fake), d_fake)
        grads_g = tg.gradient(g_loss, self.G.trainable_variables)
        self.opt_g.apply_gradients(zip(grads_g, self.G.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}
