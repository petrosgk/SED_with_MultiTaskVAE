import tensorflow as tf
import tensorflow_probability as tfp


class Model:
  def __init__(self, state_size, num_latents, num_features, num_labels,
               real_strongly_labeled_data=False, variational_encoder=False, kld_weight=None):
    self.num_features = num_features
    self.real_strongly_labeled_data = real_strongly_labeled_data
    self.variational_encoder = variational_encoder

    # Shared RNN encoder layers
    self.encoder_att_0 = tf.keras.layers.Attention(name='encoder_att_0')
    self.encoder_concat_0 = tf.keras.layers.Concatenate(axis=-1)
    self.encoder_rnn_0 = tf.keras.layers.Bidirectional(
      tf.keras.layers.GRU(units=state_size, return_sequences=True),
      name='encoder_rnn_0'
    )
    self.encoder_rnn_1 = tf.keras.layers.Bidirectional(
      tf.keras.layers.GRU(units=state_size, return_sequences=True),
      name='encoder_rnn_1'
    )
    self.encoder_dense_0 = tf.keras.layers.Dense(2 * state_size, activation='tanh', name='encoder_dense_0')
    self.encoder_add_0 = tf.keras.layers.Add()

    # Shared posterior
    self.encoder_mvn_params = tf.keras.layers.Dense(
      tfp.layers.IndependentNormal.params_size(num_latents),
      name='encoder_mvn_params'
    )
    prior = tfp.distributions.Independent(
      tfp.distributions.Normal(loc=tf.zeros(num_latents), scale=tf.ones(num_latents)),
      reinterpreted_batch_ndims=1
    )
    self.encoder_latents = tfp.layers.IndependentNormal(
      num_latents,
      activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=kld_weight),
      name='encoder_latents'
    )

    # Shared bottleneck
    self.encoder_bottleneck = tf.keras.layers.Dense(num_latents, name='encoder_bottleneck')

    # Shared RNN decoder layers
    self.decoder_att_0 = tf.keras.layers.Attention(name='decoder_att_0')
    self.decoder_concat_0 = tf.keras.layers.Concatenate(axis=-1)
    self.decoder_rnn_0 = tf.keras.layers.Bidirectional(
      tf.keras.layers.GRU(units=state_size, return_sequences=True),
      name='decoder_rnn_0'
    )
    self.decoder_rnn_1 = tf.keras.layers.Bidirectional(
      tf.keras.layers.GRU(units=state_size, return_sequences=True),
      name='decoder_rnn_1'
    )
    self.decoder_dense_0 = tf.keras.layers.Dense(2 * state_size, activation='tanh', name='decoder_dense_0')
    self.decoder_add_0 = tf.keras.layers.Add()

    # Shared output layer
    self.outputs = tf.keras.layers.Dense(num_labels, name='out')

  def encoder(self, inputs):
    att_0 = self.encoder_att_0([inputs, inputs])
    concat_0 = self.encoder_concat_0([att_0, inputs])
    rnn_0 = self.encoder_rnn_0(concat_0)
    rnn_1 = self.encoder_rnn_1(rnn_0)
    dense_0 = self.encoder_dense_0(inputs)
    add_0 = self.encoder_add_0([rnn_1, dense_0])
    if self.variational_encoder:
      mvn_params = self.encoder_mvn_params(add_0)
      latents = self.encoder_latents(mvn_params)
    else:
      latents = self.encoder_bottleneck(add_0)
    return latents

  def decoder(self, inputs, pool=False, sigmoid=False, reconstruction=False):
    att_0 = self.decoder_att_0([inputs, inputs])
    concat_0 = self.decoder_concat_0([att_0, inputs])
    rnn_0 = self.decoder_rnn_0(concat_0)
    rnn_1 = self.decoder_rnn_1(rnn_0)
    dense_0 = self.decoder_dense_0(inputs)
    add_0 = self.decoder_add_0([rnn_1, dense_0])
    if reconstruction:
      outputs = tf.keras.layers.Dense(units=self.num_features, name='rec_out')(add_0)
    else:
      if pool:
        add_0 = tf.keras.layers.GlobalAveragePooling1D()(add_0)
      outputs = self.outputs(add_0)
      if sigmoid:
        outputs = tf.keras.layers.Activation('sigmoid')(outputs)
    return outputs

  def create_model(self):
    ## Inputs
    strongly_labeled_inputs = tf.keras.layers.Input(shape=(None, self.num_features), name='strongly_labeled_inputs')
    # shape: [batch_size, sequence_length, num_features]
    weakly_labeled_inputs = tf.keras.layers.Input(shape=(None, self.num_features), name='weakly_labeled_inputs')
    # shape: [batch_size, sequence_length, num_features]
    unlabeled_inputs = tf.keras.layers.Input(shape=(None, self.num_features), name='unlabeled_inputs')
    # shape: [batch_size, sequence_length, num_features]
    sli_encoder_outputs = self.encoder(strongly_labeled_inputs)
    # shape: [batch_size, sequence_length, num_latents]
    wli_encoder_outputs = self.encoder(weakly_labeled_inputs)
    # shape: [batch_size, sequence_length, num_latents]
    uli_encoder_outputs = self.encoder(unlabeled_inputs)
    # shape: [batch_size, sequence_length, num_latents]
    sli_decoder_outputs = self.decoder(sli_encoder_outputs)
    # shape: [batch_size, sequence_length, num_labels]
    wli_decoder_outputs = self.decoder(wli_encoder_outputs, pool=True)
    # shape: [batch_size, num_labels]
    uli_decoder_outputs = self.decoder(uli_encoder_outputs, reconstruction=True)
    # shape: [batch_size, sequence_length, num_features]
    inputs = [strongly_labeled_inputs, weakly_labeled_inputs, unlabeled_inputs]
    outputs = [sli_decoder_outputs, wli_decoder_outputs, uli_decoder_outputs]
    if self.real_strongly_labeled_data:
      real_strongly_labeled_inputs = tf.keras.layers.Input(shape=(None, self.num_features), name='real_strongly_labeled_inputs')
      real_sli_encoder_outputs = self.encoder(real_strongly_labeled_inputs)
      real_sli_decoder_outputs = self.decoder(real_sli_encoder_outputs)
      inputs += [real_strongly_labeled_inputs]
      outputs += [real_sli_decoder_outputs]
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

  def create_inference_model(self):
    ## Inputs
    inputs = tf.keras.layers.Input(shape=(None, self.num_features), name='inputs')
    # shape: [batch_size, sequence_length, num_features]
    encoder_outputs = self.encoder(inputs)
    # shape: [batch_size, sequence_length, num_latents]
    decoder_outputs = self.decoder(encoder_outputs, sigmoid=True)
    # shape: [batch_size, sequence_length, num_labels]
    model = tf.keras.models.Model(inputs=inputs, outputs=decoder_outputs)
    return model
