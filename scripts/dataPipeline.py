import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


SEMILLA = 42


def buildAugmenter():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEMILLA),
        tf.keras.layers.RandomRotation(25 / 360, seed=SEMILLA, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(0.20, 0.20, seed=SEMILLA, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(0.20, seed=SEMILLA, fill_mode="nearest"),
        tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255), seed=SEMILLA),
    ])


def loadSplit(directory, img_size, batch_size, shuffle=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle,
        seed=SEMILLA,
    )
    return ds, ds.class_names


def preparar(ds, preprocess_fn, augmenter=None):
    """Aplica augmentation (solo entrenamiento) y el preprocess_input de la
    arquitectura."""
    if augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def prepararCrudo(ds, augmenter):
    ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
