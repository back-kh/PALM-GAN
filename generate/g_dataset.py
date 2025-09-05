import tensorflow as tf
import os

def load_image(path, img_size=(256,256)):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0   # normalize to [-1,1]
    return img

def make_dataset(data_dir, batch_size=32, img_size=(256,256)):
    files = tf.data.Dataset.list_files(os.path.join(data_dir, "*.jpg"), shuffle=True)
    ds = files.map(lambda p: load_image(p, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
