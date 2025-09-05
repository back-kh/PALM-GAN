# dataset.py
import os
import glob
from typing import List, Tuple, Optional, Dict
import tensorflow as tf

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
DEG_DIR_CANDIDATES = ("deg", "degraded", "input", "x")
CLEAN_DIR_CANDIDATES = ("clean", "gt", "target", "y", "bin")

def _find_subdir(root: str, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        p = os.path.join(root, c)
        if os.path.isdir(p):
            return p
    return None

def _scan_pairs(root: str) -> List[Tuple[str, Optional[str]]]:
    deg_dir = _find_subdir(root, DEG_DIR_CANDIDATES) or root
    clean_dir = _find_subdir(root, CLEAN_DIR_CANDIDATES)

    def list_images(d):
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(d, ext)))
        return sorted(files)

    x_files = list_images(deg_dir)
    if not x_files:
        raise FileNotFoundError(f"No images found under {deg_dir}")

    if clean_dir and os.path.isdir(clean_dir):
        # index by basename (without extension)
        def key(p): return os.path.splitext(os.path.basename(p))[0]
        y_files = {key(p): p for p in list_images(clean_dir)}
        pairs = []
        for xp in x_files:
            k = key(xp)
            yp = y_files.get(k, None)
            if yp is not None:
                pairs.append((xp, yp))
        if not pairs:
            raise RuntimeError(f"Found images in {deg_dir} but no matching basenames in {clean_dir}")
        return pairs
    else:
        return [(xp, None) for xp in x_files]

def _decode_grayscale(path: tf.Tensor) -> tf.Tensor:
    """Read image file and return grayscale float32 [0,1], shape [H,W,1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1], RGB
    img = tf.image.rgb_to_grayscale(img)                 # [H,W,1]
    return img

def _resize_or_crop(x: tf.Tensor, y: Optional[tf.Tensor],
                    img_size: Tuple[int, int],
                    patch_size: Optional[int],
                    training: bool) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:

    if training and patch_size is not None:
        # Ensure min size
        h = tf.shape(x)[0]; w = tf.shape(x)[1]
        ph = patch_size; pw = patch_size
        # If too small, resize up minimally before crop
        x = tf.image.resize(x, [tf.maximum(h, ph), tf.maximum(w, pw)], method="bilinear")
        if y is not None:
            y = tf.image.resize(y, [tf.maximum(h, ph), tf.maximum(w, pw)], method="nearest")
        # Random crop the same window
        if y is not None:
            stacked = tf.concat([x, y], axis=-1)
            cropped = tf.image.random_crop(stacked, size=[ph, pw, tf.shape(stacked)[-1]])
            x = cropped[..., :1]
            y = cropped[..., 1:2]
        else:
            x = tf.image.random_crop(x, size=[ph, pw, 1])
        return x, y
    else:
        x = tf.image.resize(x, img_size, method="bilinear")
        if y is not None:
            y = tf.image.resize(y, img_size, method="nearest")
        return x, y

def _augment(x: tf.Tensor, y: Optional[tf.Tensor], training: bool) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
    if not training:
        return x, y
    if tf.random.uniform(()) < 0.5:
        x = tf.image.flip_left_right(x)
        if y is not None: y = tf.image.flip_left_right(y)
    if tf.random.uniform(()) < 0.3:
        x = tf.image.flip_up_down(x)
        if y is not None: y = tf.image.flip_up_down(y)

    # Mild brightness/contrast jitter on degraded input ONLY
    if tf.random.uniform(()) < 0.5:
        x = tf.image.random_brightness(x, 0.1)
        x = tf.image.random_contrast(x, 0.9, 1.1)

    # Add small gaussian noise to input (helps robustness)
    if tf.random.uniform(()) < 0.3:
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02, dtype=x.dtype)
        x = tf.clip_by_value(x + noise, 0.0, 1.0)

    return x, y

def _load_pair(x_path: tf.Tensor, y_path: tf.Tensor,
               img_size: Tuple[int,int], patch_size: Optional[int],
               training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    x = _decode_grayscale(x_path)
    y = _decode_grayscale(y_path)
    x, y = _resize_or_crop(x, y, img_size, patch_size, training)
    x, y = _augment(x, y, training)
    # Ensure shapes known to graph
    x.set_shape([None, None, 1])
    y.set_shape([None, None, 1])
    return x, y

def _load_unpaired(x_path: tf.Tensor,
                   img_size: Tuple[int,int], patch_size: Optional[int],
                   training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    x = _decode_grayscale(x_path)
    y = tf.identity(x)
    x, y = _resize_or_crop(x, y, img_size, patch_size, training)
    x, y = _augment(x, y, training)
    x.set_shape([None, None, 1])
    y.set_shape([None, None, 1])
    return x, y

def make_dataset(
    roots: List[str],
    img_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8,
    shuffle: bool = True,
    training: bool = True,
    patch_size: Optional[int] = None,
    cache: bool = False,
) -> tf.data.Dataset:
    all_pairs: List[Tuple[str, Optional[str]]] = []
    for r in roots:
        all_pairs.extend(_scan_pairs(r))

    x_paths = []
    y_paths = []
    has_gt = True
    for xp, yp in all_pairs:
        x_paths.append(xp)
        if yp is None:
            has_gt = False
            y_paths.append(xp)  # placeholder; will be ignored in _load_unpaired
        else:
            y_paths.append(yp)

    x_ds = tf.data.Dataset.from_tensor_slices(x_paths)
    y_ds = tf.data.Dataset.from_tensor_slices(y_paths)

    ds = tf.data.Dataset.zip((x_ds, y_ds))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(4096, len(x_paths)))

    mapper = (lambda xp, yp: _load_pair(xp, yp, img_size, patch_size, training)) if has_gt \
             else (lambda xp, yp: _load_unpaired(xp, img_size, patch_size, training))

    ds = ds.map(mapper, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        ds = ds.cache()

    if training:
        ds = ds.repeat()

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def make_train_val_datasets(
    train_roots: List[str],
    val_roots: List[str],
    img_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8,
    patch_size: Optional[int] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    train_ds = make_dataset(
        train_roots, img_size=img_size, batch_size=batch_size,
        shuffle=True, training=True, patch_size=patch_size, cache=False
    )
    val_ds = make_dataset(
        val_roots, img_size=img_size, batch_size=batch_size,
        shuffle=False, training=False, patch_size=None, cache=True
    )
    return train_ds, val_ds
