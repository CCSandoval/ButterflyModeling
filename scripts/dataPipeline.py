import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


SEMILLA = 42


def aumentadorGeometrico():
    """Las cuatro caras, las que remuestrean. Van dentro del modelo, en GPU."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEMILLA, name="aug_flip"),
        tf.keras.layers.RandomRotation(25 / 360, seed=SEMILLA, fill_mode="nearest", name="aug_rot"),
        tf.keras.layers.RandomTranslation(0.20, 0.20, seed=SEMILLA, fill_mode="nearest", name="aug_tras"),
        tf.keras.layers.RandomZoom(0.20, seed=SEMILLA, fill_mode="nearest", name="aug_zoom"),
    ], name="aug_geometrica")


def aumentadorFotometrico():
    """El brillo, en tf.data: su value_range asume 0-255, antes del preprocess."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255), seed=SEMILLA),
    ], name="aug_brillo")


def buildAugmenter():
    """Las cinco en tf.data, para destilación."""
    return tf.keras.Sequential([aumentadorGeometrico(), aumentadorFotometrico()])


BUFFER_MEZCLA = 4096


def loadSplit(directory, img_size, batch_size, shuffle=True):
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=img_size,
        batch_size=None,
        label_mode="categorical",
        shuffle=False,
    )
    clases = ds.class_names
    ds = ds.map(lambda x, y: (tf.cast(x, tf.uint8), y), num_parallel_calls=AUTOTUNE).cache()
    if shuffle:
        ds = ds.shuffle(BUFFER_MEZCLA, seed=SEMILLA, reshuffle_each_iteration=True)
    return ds.batch(batch_size), clases


def preparar(ds, preprocess_fn, augmenter=None):
    """Aplica augmentation (solo entrenamiento) y el preprocess_input de la
    arquitectura."""
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if augmenter is not None:
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

