import tensorflow as tf
import numpy as np
import os
import glob
import pickle
import random
import synthesizer.model_lib as model_lib
import synthesizer.utils_lib as utils_lib
import synthesizer.callbacks_lib as callbacks_lib
import synthesizer.hparams as opt
from sklearn.model_selection import train_test_split


class BatchGenerator(tf.keras.utils.Sequence):
  def __init__(self, data_path, normalization, shuffle=False):
    self.data = glob.glob(os.path.join(data_path, '*.npz'))
    self.normalization = normalization
    self.shuffle = shuffle

  def __len__(self):
    return len(self.data)

  def load_data(self, data):
    with np.load(data) as loaded_data:
      audio_sample = loaded_data['features']
      audio_sample = self.normalization.normalize(audio_sample)
      audio_sample = np.expand_dims(audio_sample, axis=0)
    return audio_sample

  def __getitem__(self, idx):
    data = self.data[idx]
    audio_sample = self.load_data(data)
    return audio_sample, audio_sample

  def on_epoch_end(self):
    random.shuffle(self.data)


def train(audio_files, output_dir, label_per_audio_file, unlabeled_audio_files=None,
          model_name='vae', training_epochs=1000, initial_epoch=0, initial_lr=None):
  utils_lib.force_gpu_memory_growth()
  # Create training data
  data_path = os.path.join(output_dir, 'data')
  train_data_path = os.path.join(data_path, 'train')
  test_data_path = os.path.join(data_path, 'test')
  normalization_class_path = os.path.join(data_path, 'normalization.pickle')
  if opt.features == 'mel' or opt.features == 'gammatone':
    num_features = opt.num_mel_bins
  else:
    num_features = opt.n_fft // 2 + 1
  if not os.path.exists(data_path):
    os.makedirs(train_data_path, exist_ok=True)
    os.makedirs(test_data_path, exist_ok=True)
    normalization = utils_lib.Normalization()
    print('Creating training and validation data...')
    train_audio_files, test_audio_files = train_test_split(audio_files,
                                                           test_size=opt.test_size,
                                                           stratify=label_per_audio_file,
                                                           random_state=42)
    print('Extracting training data...')
    min_length, max_length = utils_lib.data_extractor(train_audio_files, train_data_path, normalization=normalization)
    print('Min. features length = %d, Max. features length = %d' % (min_length, max_length))
    print('Extracting validation data...')
    utils_lib.data_extractor(test_audio_files, test_data_path)
    with open(normalization_class_path, 'wb') as f:
      pickle.dump(normalization, f)
  else:
    print('Reusing data found in %s' % data_path)
    with open(normalization_class_path, 'rb') as f:
      normalization = pickle.load(f)
  checkpoint_path = os.path.join(output_dir, 'checkpoints')
  os.makedirs(checkpoint_path, exist_ok=True)
  model_path = os.path.join(checkpoint_path, '{}.h5'.format(model_name))
  # Create model
  vae = model_lib.Model(state_size=opt.state_size,
                        num_latents=opt.num_latents,
                        num_features=num_features,
                        kld_weight=opt.kld_weight)
  keras_model = vae.create_model()
  learning_rate = opt.learning_rate if initial_lr is None else initial_lr
  keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.losses.MeanSquaredError(name='mse')])
  if os.path.exists(model_path):
    print('Loading model weights from: %s' % model_path)
    keras_model.load_weights(filepath=model_path)
  else:
    print('No model weights found in %s. Starting new training...' % model_path)
  print(keras_model.summary())
  # Create batch data generator
  batch_generator = BatchGenerator(train_data_path, normalization=normalization, shuffle=True)
  # Create batch data generator for validation data
  val_batch_generator = BatchGenerator(test_data_path, normalization=normalization)
  # Define callbacks
  logs_path = os.path.join(output_dir, 'logs', '{}'.format(model_name))
  os.makedirs(logs_path, exist_ok=True)
  outputs_path = os.path.join(output_dir, 'outputs', '{}'.format(model_name))
  callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                  monitor='val_loss',
                                                  save_weights_only=True,
                                                  save_best_only=True,
                                                  verbose=1),
               tf.keras.callbacks.TensorBoard(log_dir=logs_path),
               callbacks_lib.VAECallback(data_path=test_data_path,
                                         model_path=model_path,
                                         save_path=outputs_path,
                                         normalization=normalization,
                                         infer_freq_epochs=opt.infer_freq_epochs,
                                         num_iterations=opt.num_iterations,
                                         num_perturbations=opt.num_perturbations)
               ]
  # Fit model
  history = keras_model.fit(x=batch_generator,
                            validation_data=val_batch_generator,
                            epochs=training_epochs,
                            callbacks=callbacks,
                            verbose=1,
                            initial_epoch=initial_epoch)
  print('Finished training.')
  return history
