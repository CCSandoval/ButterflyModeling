from tensorflow.keras import Model, layers
from tensorflow.keras.applications import (
    ConvNeXtTiny,
    DenseNet121,
    EfficientNetV2B0,
    MobileNetV3Small,
    ResNet50V2,
)
from tensorflow.keras.applications.convnext import preprocess_input as convnextPreprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenetPreprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnetPreprocess
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetPreprocess
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetPreprocess

# nombre -> (clase base, preprocess_input, capa conv para Grad-CAM)
ARCHITECTURES = {
    "mobilenet_v3_small": (MobileNetV3Small, mobilenetPreprocess, "conv_1"),
    "efficientnet_v2_b0": (EfficientNetV2B0, efficientnetPreprocess, "top_activation"),
    "resnet50_v2": (ResNet50V2, resnetPreprocess, "post_relu"),
    "densenet121": (DenseNet121, densenetPreprocess, "relu"),
    "convnext_tiny": (ConvNeXtTiny, convnextPreprocess, "convnext_tiny_stage_3_block_2_identity"),
}


IMG_SIZE = (320, 320)


def buildModel(name, numClasses):
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
    """Student pequeño para destilación: sin pesos preentrenados, pocas capas."""
    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)
    for filters in (32, 64, 128, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.30)(x)
    output = layers.Dense(numClasses, activation="softmax")(x)
    return Model(inputs=inputs, outputs=output)
