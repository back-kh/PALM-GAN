import tensorflow as tf
from ..models.generator import build_generator
from ..models.discriminator import build_discriminator
from ..losses.adversarial import lsgan_d, lsgan_g
from ..losses.reconstruction import bce_loss, ssim_loss

class Trainer:
    def __init__(self, cfg):
        self.G = build_generator(attn_at=tuple(cfg["model"]["gen"]["attn_at"]))
        self.D = build_discriminator()
        self.opt_g = tf.keras.optimizers.Adam(cfg["optim"]["lr_g"], beta_1=cfg["optim"]["betas"][0], beta_2=cfg["optim"]["betas"][1])
        self.opt_d = tf.keras.optimizers.Adam(cfg["optim"]["lr_d"], beta_1=cfg["optim"]["betas"][0], beta_2=cfg["optim"]["betas"][1])
        self.cfg = cfg

    @tf.function
    def train_step(self, x, y):
        # --- update D ---
        with tf.GradientTape() as td:
            y_fake = self.G(x, training=True)
            d_real = self.D([x, y], training=True)
            d_fake = self.D([x, y_fake], training=True)
            ld = lsgan_d(d_real, d_fake)
        self.opt_d.apply_gradients(zip(td.gradient(ld, self.D.trainable_variables), self.D.trainable_variables))

        # --- update G ---
        with tf.GradientTape() as tg:
            y_fake = self.G(x, training=True)
            d_fake = self.D([x, y_fake], training=False)
            l_adv = lsgan_g(d_fake)
            l_rec = bce_loss(y, y_fake)  # or L1
            l_ssim = ssim_loss(y, y_fake)
            lg = self.cfg["loss"]["adv_weight"]*l_adv + \
                 self.cfg["loss"]["recon"]["weight"]*l_rec + \
                 self.cfg["loss"]["ssim_weight"]*l_ssim
        self.opt_g.apply_gradients(zip(tg.gradient(lg, self.G.trainable_variables), self.G.trainable_variables))
        return {"ld": ld, "lg": lg, "adv": l_adv, "rec": l_rec, "ssim": l_ssim}
