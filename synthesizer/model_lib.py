import tensorflow as tf
import tensorflow_probability as tfp


class Model:
  def __init__(self, state_size, num_latents, num_features, kld_weight=None,
               use_unlabeled_data=False):
    self.num_latents = num_latents
    self.num_features = num_features
    self.use_unlabeled_data = use_unlabeled_data

    # RNN encoder layers
    self.encoder_rnn_0 = tf.keras.layers.Bidirectional(
      tf.keras.layers.GRU(units=state_size, return_state=True, return_sequences=True),
      name='encoder_rnn_0'
    )
    self.encoder_rnn_1 = tf.keras.layers.Bidirectional(
      tf.keras.layers.GRU(units=state_size, return_sequences=True),
      name='encoder_rnn_1'
    )

    # Posterior
    self.encoder_mvn_params = tf.keras.layers.Dense(
      tfp.layers.MultivariateNormalTriL.params_size(num_latents),
      name='encoder_mvn_params'
    )
    self.encoder_latents = tfp.layers.MultivariateNormalTriL(
      num_latents,
      activity_regularizer=tfp.layers.KLDivergenceRegularizer(
        tfp.distributions.MultivariateNormalTriL(loc=tf.zeros(num_latents)),
        weight=kld_weight
      ),
      name='encoder_latents'
    )

    # RNN decoder layers
    self.decoder_rnn_0 = tf.keras.layers.Bidirectional(
      tf.keras.layers.GRU(units=state_size, return_state=True, return_sequences=True),
      name='decoder_rnn_0'
    )
    self.decoder_rnn_1 = tf.keras.layers.Bidirectional(
      tf.keras.layers.GRU(units=state_size, return_sequences=True),
      name='decoder_rnn_1'
    )

    # Shared output layer
    self.outputs = tf.keras.layers.Dense(num_features, name='decoder_out')

  def encoder(self, inputs):
    rnn_0 = self.encoder_rnn_0(inputs)
    rnn_1 = self.encoder_rnn_1(rnn_0)
    mvn_params = self.encoder_mvn_params(rnn_1)
    latents = self.encoder_latents(mvn_params)
    return latents, mvn_params

  def decoder(self, inputs):
    rnn_0 = self.decoder_rnn_0(inputs)
    rnn_1 = self.decoder_rnn_1(rnn_0)
    outputs = self.outputs(rnn_1)
    return outputs

  def create_model(self):
    ## Main reconstruction task inputs
    main_inputs = tf.keras.layers.Input(shape=(None, self.num_features), name='main_inputs')
    # shape: [batch_size, sequence_length, num_features]
    latents, _ = self.encoder(main_inputs)
    # shape: [batch_size, sequence_length, num_latents]
    decoder_outputs = self.decoder(latents)
    # shape: [batch_size, sequence_length, num_features]
    inputs = [main_inputs]
    outputs = [decoder_outputs]
    if self.use_unlabeled_data:
      ## Unlabeled reconstruction task inputs
      unlabeled_inputs = tf.keras.layers.Input(shape=(None, self.num_features), name='unlabeled_inputs')
      # shape: [batch_size, sequence_length, num_features]
      uli_latents, _ = self.encoder(unlabeled_inputs)
      # shape: [batch_size, sequence_length, num_latents]
      uli_decoder_outputs = self.decoder(uli_latents)
      # shape: [batch_size, sequence_length, num_features]
      inputs += [unlabeled_inputs]
      outputs += [uli_decoder_outputs]
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

  def create_encoder_inference_model(self):
    ## Inputs
    inputs = tf.keras.layers.Input(shape=(None, self.num_features), name='inputs')
    # shape: [batch_size, sequence_length, num_features]
    _, mvn_params = self.encoder(inputs)
    # shape: [batch_size, sequence_length, params_size]
    model = tf.keras.models.Model(inputs=inputs, outputs=mvn_params)
    return model

  def create_decoder_inference_model(self):
    ## Inputs
    inputs = tf.keras.layers.Input(shape=(None, self.num_latents), name='inputs')
    # shape: [batch_size, sequence_length, num_latents]
    decoder_outputs = self.decoder(inputs)
    # shape: [batch_size, sequence_length, num_features]
    model = tf.keras.models.Model(inputs=inputs, outputs=decoder_outputs)
    return model
