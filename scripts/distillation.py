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
    """
    Hint loss con las features normalizadas por posición.
    Sin normalizar, el Mean Squared Error crece por la diferencia de escala entre arquitecturas
    """
    proyectada = proyeccion(featuresEstudiante)
    if proyectada.shape[1:3] != featuresDocente.shape[1:3]:
        proyectada = tf.image.resize(proyectada, featuresDocente.shape[1:3])
    return tf.reduce_mean(tf.square(tf.math.l2_normalize(proyectada, axis=-1)
                                    - tf.math.l2_normalize(featuresDocente, axis=-1)))


def mapaAtencion(features):
    """Suma de cuadrados sobre canales, aplanada y normalizada."""
    mapa = tf.reduce_sum(tf.square(features), axis=-1)
    return tf.math.l2_normalize(tf.reshape(mapa, [tf.shape(mapa)[0], -1]), axis=1)


def perdidaAtencion(featuresDocente, featuresEstudiante):
    """Attention transfer: colapsa los canales, así que no necesita proyección
    y no importa la escala"""
    if featuresEstudiante.shape[1:3] != featuresDocente.shape[1:3]:
        featuresEstudiante = tf.image.resize(featuresEstudiante, featuresDocente.shape[1:3])
    return tf.reduce_mean(tf.square(mapaAtencion(featuresDocente)
                                    - mapaAtencion(featuresEstudiante)))


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

        # solo la clase concreta construye: una subclase todavía tiene que
        # crear sus propios submodelos, y Keras prohíbe añadirlos después
        if type(self) is Destilador:
            self.construir()

    def construir(self):
        """Un modelo subclaseado no queda construido hasta que se lo llama, y
        BackupAndRestore lo exige construido antes de `fit()`. Se llama al
        final del `__init__` de cada clase concreta."""
        self(tf.zeros((1, *self.estudiante.input_shape[1:])))

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


class DestiladorFeatures(Destilador):
    """
    KD de respuesta más hint loss entre features normalizadas.
    Toma la capa previa al pool de cada modelo, que es post-activación en toda arquitectura
    """

    NOMBRE_AUXILIAR = "loss_features"
    USA_PROYECCION = True

    def __init__(self, docente, estudiante, preprocessDocente, preprocessEstudiante,
                 temperatura, alfa, beta):
        super().__init__(docente, estudiante, preprocessDocente, preprocessEstudiante,
                         temperatura, alfa)
        from .architectures import capaFeatures

        self.beta = beta
        self.docenteDual = tf.keras.Model(docente.input, [capaFeatures(docente), docente.output])
        self.docenteDual.trainable = False
        self.estudianteDual = tf.keras.Model(
            estudiante.input, [capaFeatures(estudiante), estudiante.output])

        self.proyeccion = None
        if self.USA_PROYECCION:
            self.proyeccion = construirProyeccion(self.docenteDual.output[0].shape[-1])
            self.proyeccion.build(self.estudianteDual.output[0].shape)
        self.metricaAuxiliar = tf.keras.metrics.Mean(name=self.NOMBRE_AUXILIAR)

        self.construir()

    @property
    def metrics(self):
        return [self.metricaPerdida, self.metricaAccuracy, self.metricaAuxiliar]

    def perdidaAuxiliar(self, featuresDocente, featuresEstudiante):
        return perdidaFeatures(featuresDocente, featuresEstudiante, self.proyeccion)

    def entrenables(self):
        return self.estudiante.trainable_variables + self.proyeccion.trainable_variables

    def train_step(self, data):
        x, y, pesos = tf.keras.utils.unpack_x_y_sample_weight(data)
        featuresDocente, probsDocente = self.docenteDual(
            self.preprocessDocente(x), training=False)

        with tf.GradientTape() as cinta:
            featuresEstudiante, probsEstudiante = self.estudianteDual(
                self.preprocessEstudiante(x), training=True)
            respuesta = perdidaRespuesta(
                probsDocente, probsEstudiante, y, self.temperatura, self.alfa, pesos)
            auxiliar = self.perdidaAuxiliar(featuresDocente, featuresEstudiante)
            perdida = respuesta + self.beta * auxiliar

        entrenables = self.entrenables()
        self.optimizer.apply_gradients(zip(cinta.gradient(perdida, entrenables), entrenables))

        self.metricaPerdida.update_state(perdida)
        self.metricaAccuracy.update_state(y, probsEstudiante)
        self.metricaAuxiliar.update_state(auxiliar)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        resultado = super().test_step(data)
        self.metricaAuxiliar.update_state(0.0)
        return resultado


class DestiladorAtencion(DestiladorFeatures):
    """
    KD de respuesta más attention transfer.
    """

    NOMBRE_AUXILIAR = "loss_atencion"
    USA_PROYECCION = False

    def perdidaAuxiliar(self, featuresDocente, featuresEstudiante):
        return perdidaAtencion(featuresDocente, featuresEstudiante)

    def entrenables(self):
        return self.estudiante.trainable_variables
