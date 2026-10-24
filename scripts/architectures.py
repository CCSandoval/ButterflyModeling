import numpy as np
from tensorflow.keras import Model, layers

from scripts.dataPipeline import aumentadorGeometrico
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB0,
    EfficientNetV2B0,
    EfficientNetV2S,
    InceptionV3,
    MobileNetV2,
    MobileNetV3Small,
    NASNetMobile,
    ResNet50V2,
)
from tensorflow.keras.applications.densenet import preprocess_input as densenetPreprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnetPreprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetV2Preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionPreprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetV2Preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetV3Preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnetPreprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetPreprocess


# nombre interno -> como se escribe en el artículo y en la interfaz
NOMBRES = {
    "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNetB0",
    "efficientnet_v2_b0": "EfficientNetV2B0",
    "efficientnet_v2_s": "EfficientNetV2S",
    "inception_v3": "InceptionV3",
    "mobilenet_v2": "MobileNetV2",
    "mobilenet_v3_small": "MobileNetV3Small",
    "nasnet_mobile": "NASNetMobile",
    "resnet50_v2": "ResNet50V2",
}

# nombre -> (clase base, preprocess_input)
ARCHITECTURES = {
    "efficientnet_v2_s": (EfficientNetV2S, efficientnetV2Preprocess),
    "efficientnet_v2_b0": (EfficientNetV2B0, efficientnetV2Preprocess),
    "resnet50_v2": (ResNet50V2, resnetPreprocess),
    "densenet121": (DenseNet121, densenetPreprocess),
    "inception_v3": (InceptionV3, inceptionPreprocess),
    "mobilenet_v3_small": (MobileNetV3Small, mobilenetV3Preprocess),
    "mobilenet_v2": (MobileNetV2, mobilenetV2Preprocess),
    "efficientnet_b0": (EfficientNetB0, efficientnetPreprocess),
    "nasnet_mobile": (NASNetMobile, nasnetPreprocess),
}

IMG_SIZE = (448, 448)
CABEZA_POOL = "cabeza_pool"


def buildModel(name, numClasses, aumentar=True):
    """`aumentar=False` para destilación, que aumenta en tf.data."""
    baseClass, _ = ARCHITECTURES[name]
    entrada = layers.Input(shape=(*IMG_SIZE, 3))
    tensor = aumentadorGeometrico()(entrada) if aumentar else entrada
    base = baseClass(weights="imagenet", include_top=False, input_tensor=tensor)
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D(name=CABEZA_POOL)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.50)(x)
    output = layers.Dense(numClasses, activation="softmax")(x)

    return Model(inputs=entrada, outputs=output)


def descongelarCola(modelo, fraccion):
    """Deja entrenable la última `fraccion` de capas del backbone.

    Las BatchNormalization quedan congeladas a propósito: con lotes de 32 sus
    estadísticas de ImageNet se degradan más de lo que aporta ajustarlas.
    """
    corte = [l.name for l in modelo.layers].index(CABEZA_POOL)
    backbone = [c for c in modelo.layers[:corte] if not c.name.startswith("aug_")]
    desde = int(len(backbone) * (1 - fraccion))

    for capa in backbone[desde:]:
        if not isinstance(capa, layers.BatchNormalization):
            capa.trainable = True

    return {
        "capas_backbone": len(backbone),
        "capas_descongeladas": sum(1 for c in backbone[desde:] if c.trainable),
        "parametros_entrenables": int(sum(np.prod(w.shape) for w in modelo.trainable_weights)),
    }


def capaFeatures(modelo):
    """Última salida del backbone"""
    return modelo.get_layer(CABEZA_POOL).input


def preprocessFn(name):
    return ARCHITECTURES[name][1]
