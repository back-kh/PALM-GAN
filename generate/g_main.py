import tensorflow as tf
from dataset import make_dataset
from trainer import DCGANTrainer
import os

def main():
    data_dir = "data/"
    batch_size = 16
    epochs = 50
    latent_dim = 128
    img_shape = (256,256,3)

    ds = make_dataset(data_dir, batch_size=batch_size, img_size=img_shape[:2])
    trainer = DCGANTrainer(latent_dim, img_shape, lr=2e-4)

    for epoch in range(epochs):
        for step, real_imgs in enumerate(ds):
            logs = trainer.train_step(real_imgs)
            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step} D_loss={logs['d_loss']:.4f} G_loss={logs['g_loss']:.4f}")

        # save sample image
        noise = tf.random.normal([4, latent_dim])
        fake = trainer.G(noise, training=False)
        fake = (fake + 1.0) * 127.5
        fake = tf.cast(fake, tf.uint8)
        os.makedirs("samples", exist_ok=True)
        for i, img in enumerate(fake):
            tf.keras.preprocessing.image.save_img(f"samples/epoch{epoch}_sample{i}.png", img)

if __name__ == "__main__":
    main()
