import tensorflow as tf
import tensorflow.python.keras.backend as K


class F1Score(tf.losses.Loss):
  def __init__(self, soft_f1_score=False, from_logits=False, is_sequence=False, threshold=0.5, name='f1_score'):
    super(F1Score, self).__init__(name=name)
    self.from_logits = from_logits
    self.is_sequence = is_sequence
    self.soft_f1_score = soft_f1_score
    self.threshold = threshold

  def call(self, y_true, y_pred):
    if self.from_logits:
      y_pred = tf.nn.sigmoid(y_pred)
    if not self.soft_f1_score:
      y_pred = y_pred >= self.threshold
    y_true = tf.cast(y_true, 'float32')
    y_pred = tf.cast(y_pred, 'float32')
    axis = [0, 1] if self.is_sequence else 0
    tp = tf.reduce_sum(y_true * y_pred, axis=axis)
    tn = tf.reduce_sum((1. - y_true) * (1. - y_pred), axis=axis)
    fp = tf.reduce_sum((1. - y_true) * y_pred, axis=axis)
    fn = tf.reduce_sum(y_true * (1. - y_pred), axis=axis)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2. * p * r / (p + r + K.epsilon())
    loss = tf.reduce_mean(f1)
    if not self.soft_f1_score:
      return loss
    else:
      return 1. - loss
