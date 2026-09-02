import tensorflow as tf


def temperarProbs(probs, temperatura):
    logProbs = tf.math.log(tf.clip_by_value(probs, 1e-8, 1.0))
    return tf.nn.softmax(logProbs / temperatura)


def perdidaRespuesta(probsDocente, probsEstudiante, etiquetas, temperatura=4.0, alfa=0.7):
    blandoDocente = temperarProbs(probsDocente, temperatura)
    blandoEstudiante = temperarProbs(probsEstudiante, temperatura)
    kl = tf.keras.losses.KLDivergence()(blandoDocente, blandoEstudiante)
    duro = tf.keras.losses.CategoricalCrossentropy()(etiquetas, probsEstudiante)
    return alfa * (temperatura ** 2) * kl + (1.0 - alfa) * duro


def construirProyeccion(canalesSalida):
    return tf.keras.layers.Conv2D(canalesSalida, 1, padding="same")


def perdidaFeatures(featuresDocente, featuresEstudiante, proyeccion):
    proyectada = proyeccion(featuresEstudiante)
    if proyectada.shape[1:3] != featuresDocente.shape[1:3]:
        proyectada = tf.image.resize(proyectada, featuresDocente.shape[1:3])
    return tf.reduce_mean(tf.square(proyectada - featuresDocente))
