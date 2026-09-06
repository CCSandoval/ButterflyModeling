from tensorflow.keras import Model, layers
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB0,
    EfficientNetV2B0,
    EfficientNetV2S,
    InceptionV3,
    MobileNet,
    MobileNetV2,
    MobileNetV3Large,
    MobileNetV3Small,
    NASNetMobile,
    ResNet50V2,
)
from tensorflow.keras.applications.densenet import preprocess_input as densenetPreprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnetPreprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetV2Preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionPreprocess
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenetPreprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetV2Preprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetV3Preprocess
from tensorflow.keras.applications.nasnet import preprocess_input as nasnetPreprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetPreprocess


def sinPreprocess(x):
    """La CNN compacta lleva su propio Rescaling, así que recibe 0-255 tal cual."""
    return x


# nombre -> (clase base, preprocess_input, capa conv para Grad-CAM)
ARCHITECTURES = {
    "efficientnet_v2_s": (EfficientNetV2S, efficientnetV2Preprocess, "top_activation"),
    "efficientnet_v2_b0": (EfficientNetV2B0, efficientnetV2Preprocess, "top_activation"),
    "resnet50_v2": (ResNet50V2, resnetPreprocess, "post_relu"),
    "densenet121": (DenseNet121, densenetPreprocess, "relu"),
    "inception_v3": (InceptionV3, inceptionPreprocess, "mixed10"),
    "mobilenet_v3_small": (MobileNetV3Small, mobilenetV3Preprocess, "conv_1"),
    "mobilenet_v2": (MobileNetV2, mobilenetV2Preprocess, "out_relu"),
    "mobilenet": (MobileNet, mobilenetPreprocess, "conv_pw_13_relu"),
    "mobilenet_v3_large": (MobileNetV3Large, mobilenetV3Preprocess, "conv_1"),
    "efficientnet_b0": (EfficientNetB0, efficientnetPreprocess, "top_activation"),
    "nasnet_mobile": (NASNetMobile, nasnetPreprocess, "normal_concat_12"),
    "cnn_compacta": (None, sinPreprocess, "conv_final"),
}

# Candidatos a docente: se elige el de mejor macro-F1, sin mirar costo.
DOCENTES = [
    "efficientnet_v2_s",
    "efficientnet_v2_b0",
    "resnet50_v2",
    "densenet121",
    "inception_v3",
]

# Candidatos a estudiante: se elige por costo/calidad, no por calidad sola.
ESTUDIANTES = [
    "mobilenet_v3_small",   # separable + SE
    "mobilenet_v2",         # residual invertido
    "efficientnet_b0",      # escalado compuesto
    "nasnet_mobile",        # búsqueda automática de arquitectura
]

IMG_SIZE = (320, 320)


def buildModel(name, numClasses):
    if name == "cnn_compacta":
        return buildCompactCNN(numClasses)

    baseClass, _, _ = ARCHITECTURES[name]
    base = baseClass(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
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

    return Model(inputs=base.input, outputs=output)


def preprocessFn(name):
    return ARCHITECTURES[name][1]


def gradcamLayer(name):
    return ARCHITECTURES[name][2]


def buildCompactCNN(numClasses):
    """Sin pesos preentrenados: es el piso de la comparación de estudiantes,
    mide cuánto del desempeño viene de ImageNet y no de la arquitectura."""
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    bloques = (32, 64, 128, 128)
    for i, filters in enumerate(bloques):
        ultimo = i == len(bloques) - 1
        x = layers.Conv2D(filters, 3, strides=2 if i == 0 else 1, padding="same",
                          activation="relu",
                          name="conv_final" if ultimo else f"conv_{i}")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    output = layers.Dense(numClasses, activation="softmax")(x)
    return Model(inputs=inputs, outputs=output)
