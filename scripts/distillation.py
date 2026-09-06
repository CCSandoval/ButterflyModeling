import tensorflow as tf


def temperarProbs(probs, temperatura):
    logProbs = tf.math.log(tf.clip_by_value(probs, 1e-8, 1.0))
    return tf.nn.softmax(logProbs / temperatura)


def perdidaRespuesta(probsDocente, probsEstudiante, etiquetas, temperatura, alfa, pesos=None):
    blandoDocente = temperarProbs(probsDocente, temperatura)
    blandoEstudiante = temperarProbs(probsEstudiante, temperatura)
    kl = tf.reduce_sum(
        blandoDocente * (tf.math.log(blandoDocente + 1e-9) - tf.math.log(blandoEstudiante + 1e-9)),
        axis=-1,
    )
    duro = tf.keras.losses.categorical_crossentropy(etiquetas, probsEstudiante)
    porMuestra = alfa * (temperatura ** 2) * kl + (1.0 - alfa) * duro
    if pesos is not None:
        porMuestra = porMuestra * tf.cast(pesos, porMuestra.dtype)
    return tf.reduce_mean(porMuestra)


def construirProyeccion(canalesSalida):
    return tf.keras.layers.Conv2D(canalesSalida, 1, padding="same")


def perdidaFeatures(featuresDocente, featuresEstudiante, proyeccion):
    proyectada = proyeccion(featuresEstudiante)
    if proyectada.shape[1:3] != featuresDocente.shape[1:3]:
        proyectada = tf.image.resize(proyectada, featuresDocente.shape[1:3])
    return tf.reduce_mean(tf.square(proyectada - featuresDocente))


class Destilador(tf.keras.Model):
    """Entrena al estudiante con KD de respuesta. Recibe imágenes crudas (0-255)
    y aplica el preprocess de cada modelo por dentro, de forma que ambos vean
    exactamente la misma vista aumentada.

    `val_loss` es la CE del estudiante, no la pérdida de destilación: así el
    EarlyStopping monitorea lo mismo que en los notebooks 01-05 y los modelos
    quedan comparables.
    """

    def __init__(self, docente, estudiante, preprocessDocente, preprocessEstudiante,
                 temperatura, alfa):
        super().__init__()
        self.docente = docente
        self.estudiante = estudiante
        self.preprocessDocente = preprocessDocente
        self.preprocessEstudiante = preprocessEstudiante
        self.temperatura = temperatura
        self.alfa = alfa
        self.docente.trainable = False

        self.metricaPerdida = tf.keras.metrics.Mean(name="loss")
        self.metricaAccuracy = tf.keras.metrics.CategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.metricaPerdida, self.metricaAccuracy]

    def call(self, x, training=False):
        return self.estudiante(self.preprocessEstudiante(x), training=training)

    def train_step(self, data):
        x, y, pesos = tf.keras.utils.unpack_x_y_sample_weight(data)
        probsDocente = self.docente(self.preprocessDocente(x), training=False)

        with tf.GradientTape() as cinta:
            probsEstudiante = self.estudiante(self.preprocessEstudiante(x), training=True)
            perdida = perdidaRespuesta(
                probsDocente, probsEstudiante, y, self.temperatura, self.alfa, pesos
            )

        entrenables = self.estudiante.trainable_variables
        self.optimizer.apply_gradients(zip(cinta.gradient(perdida, entrenables), entrenables))

        self.metricaPerdida.update_state(perdida)
        self.metricaAccuracy.update_state(y, probsEstudiante)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        probsEstudiante = self.estudiante(self.preprocessEstudiante(x), training=False)

        self.metricaPerdida.update_state(
            tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y, probsEstudiante))
        )
        self.metricaAccuracy.update_state(y, probsEstudiante)
        return {m.name: m.result() for m in self.metrics}
