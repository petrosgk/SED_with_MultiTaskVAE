import tensorflow as tf
import numpy as np
import os
import glob
import math
import random
import pickle
import event_detector.model_lib as model_lib
import event_detector.losses_lib as losses_lib
import event_detector.utils_lib as utils_lib
import event_detector.hparams as opt
from sklearn.model_selection import train_test_split


class BatchGenerator(tf.keras.utils.Sequence):
  def __init__(self, strongly_labeled_data_path, weakly_labeled_data_path, unlabeled_data_path, batch_size, normalization,
               real_strongly_labeled_data_path=None):
    self.strongly_labeled_data = glob.glob(os.path.join(strongly_labeled_data_path, '*.npz'))
    self.weakly_labeled_data = glob.glob(os.path.join(weakly_labeled_data_path, '*.npz'))
    self.unlabeled_data = glob.glob(os.path.join(unlabeled_data_path, '*.npz'))
    self.batch_size = batch_size
    self.normalization = normalization
    self.real_strongly_labeled_data = None
    if real_strongly_labeled_data_path is not None:
      self.real_strongly_labeled_data = glob.glob(os.path.join(real_strongly_labeled_data_path, '*.npz'))

  def __len__(self):
    return math.ceil(len(self.strongly_labeled_data) / self.batch_size)

  def create_training_sample(self, data, load_mask=False, pad_mask=False):
    data_batch = random.sample(data, k=self.batch_size)
    audio_sample_batch = []
    mask_batch = []
    for data_sample in data_batch:
      with np.load(data_sample) as loaded_data_sample:
        audio_sample = loaded_data_sample['features']
        audio_sample = self.normalization.normalize(audio_sample)
        audio_sample_batch.append(audio_sample)
        if load_mask:
          mask = loaded_data_sample['mask']
          mask_batch.append(mask)
    audio_sample_batch = tf.keras.preprocessing.sequence.pad_sequences(audio_sample_batch, dtype='float32', padding='post')
    if load_mask:
      if pad_mask:
        mask_batch = tf.keras.preprocessing.sequence.pad_sequences(mask_batch, dtype='float32', padding='post')
      else:
        mask_batch = np.asarray(mask_batch)
      return audio_sample_batch, mask_batch
    else:
      return audio_sample_batch

  def __getitem__(self, idx):
    strongly_labeled_audio_sample_batch, strongly_labeled_mask_batch = self.create_training_sample(
      self.strongly_labeled_data, load_mask=True, pad_mask=True
    )
    weakly_labeled_audio_sample_batch, weakly_labeled_mask_batch = self.create_training_sample(
      self.weakly_labeled_data, load_mask=True
    )
    unlabeled_audio_sample_batch = self.create_training_sample(self.unlabeled_data)
    inputs = [strongly_labeled_audio_sample_batch, weakly_labeled_audio_sample_batch, unlabeled_audio_sample_batch]
    targets = [strongly_labeled_mask_batch, weakly_labeled_mask_batch, unlabeled_audio_sample_batch]
    if self.real_strongly_labeled_data is not None:
      real_strongly_labeled_audio_sample_batch, real_strongly_labeled_mask_batch = self.create_training_sample(
        self.real_strongly_labeled_data, load_mask=True, pad_mask=True
      )
      inputs += [real_strongly_labeled_audio_sample_batch]
      targets += [real_strongly_labeled_mask_batch]
    return inputs, targets


def train(strongly_labeled_audio_files, weakly_labeled_audio_files, unlabeled_audio_files, test_audio_files, output_dir,
          real_strongly_labeled_audio_files=None, model_name='ead', training_epochs=1000, initial_epoch=0, initial_lr=None):
  utils_lib.force_gpu_memory_growth()
  # Create training data
  data_path = os.path.join(output_dir, 'data')
  train_data_path = os.path.join(data_path, 'train')
  strongly_labeled_data_path = os.path.join(train_data_path, 'strong')
  weakly_labeled_data_path = os.path.join(train_data_path, 'weak')
  unlabeled_data_path = os.path.join(train_data_path, 'unlabeled')
  test_data_path = os.path.join(data_path, 'test')
  strongly_labeled_test_data_path = os.path.join(test_data_path, 'strong')
  weakly_labeled_test_data_path = os.path.join(test_data_path, 'weak')
  unlabeled_test_data_path = os.path.join(test_data_path, 'unlabeled')
  real_strongly_labeled_data_path = None
  real_strongly_labeled_test_data_path = None
  if real_strongly_labeled_audio_files is not None:
    real_strongly_labeled_data_path = os.path.join(train_data_path, 'strong_real')
    real_strongly_labeled_test_data_path = os.path.join(test_data_path, 'strong_real')
  normalization_class_path = os.path.join(data_path, 'normalization.pickle')
  if opt.features == 'mel' or opt.features == 'gammatone':
    num_features = opt.num_mel_bins
  else:
    num_features = opt.n_fft // 2 + 1
  if not os.path.exists(data_path):
    os.makedirs(strongly_labeled_data_path, exist_ok=True)
    os.makedirs(weakly_labeled_data_path, exist_ok=True)
    os.makedirs(unlabeled_data_path, exist_ok=True)
    os.makedirs(strongly_labeled_test_data_path, exist_ok=True)
    os.makedirs(weakly_labeled_test_data_path, exist_ok=True)
    os.makedirs(unlabeled_test_data_path, exist_ok=True)
    normalization = utils_lib.Normalization()
    print('Creating training and validation data...')
    weakly_labeled_train_audio_files, weakly_labeled_test_audio_files = train_test_split(
      weakly_labeled_audio_files, test_size=opt.test_size, random_state=42
    )
    unlabeled_train_audio_files, unlabeled_test_audio_files = train_test_split(
      unlabeled_audio_files, test_size=opt.test_size, random_state=42
    )
    print('Extracting strongly labeled training data...')
    utils_lib.data_extractor(strongly_labeled_audio_files, strongly_labeled_data_path, normalization=normalization)
    print('Extracting weakly labeled training data...')
    utils_lib.data_extractor(weakly_labeled_train_audio_files, weakly_labeled_data_path, normalization=normalization, weak_labels=True)
    print('Extracting unlabeled training data...')
    utils_lib.data_extractor(unlabeled_train_audio_files, unlabeled_data_path, normalization=normalization, unlabeled=True)
    print('Extracting strongly labeled validation data...')
    utils_lib.data_extractor(test_audio_files, strongly_labeled_test_data_path)
    print('Extracting weakly labeled validation data...')
    utils_lib.data_extractor(weakly_labeled_test_audio_files, weakly_labeled_test_data_path, weak_labels=True)
    print('Extracting unlabeled validation data...')
    utils_lib.data_extractor(unlabeled_test_audio_files, unlabeled_test_data_path, unlabeled=True)
    if real_strongly_labeled_audio_files is not None:
      os.makedirs(real_strongly_labeled_data_path, exist_ok=True)
      os.makedirs(real_strongly_labeled_test_data_path, exist_ok=True)
      real_strongly_labeled_train_audio_files, real_strongly_labeled_test_audio_files = train_test_split(
        real_strongly_labeled_audio_files, test_size=opt.test_size, random_state=42
      )
      print('Extracting real strongly labeled training data...')
      utils_lib.data_extractor(real_strongly_labeled_train_audio_files, real_strongly_labeled_data_path, normalization=normalization)
      print('Extracting real strongly labeled validation data...')
      utils_lib.data_extractor(real_strongly_labeled_test_audio_files, real_strongly_labeled_test_data_path)
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
  real_strongly_labeled_data = True if real_strongly_labeled_audio_files is not None else False
  ead = model_lib.Model(state_size=opt.state_size,
                        num_latents=opt.num_latents,
                        variational_encoder=opt.variational_encoder,
                        kld_weight=opt.kld_weight,
                        num_features=num_features,
                        num_labels=len(opt.labels),
                        real_strongly_labeled_data=real_strongly_labeled_data)
  keras_model = ead.create_model()
  learning_rate = opt.learning_rate if initial_lr is None else initial_lr
  loss = [tf.keras.losses.BinaryCrossentropy(from_logits=True),
          tf.keras.losses.BinaryCrossentropy(from_logits=True),
          tf.keras.losses.MeanSquaredError()]
  metrics = [[losses_lib.F1Score(from_logits=True, is_sequence=True)],
             [losses_lib.F1Score(from_logits=True)],
             []]
  loss_weights = [opt.synthetic_loss_weight, 1.0, 1.0]
  if real_strongly_labeled_audio_files is not None:
    loss += [tf.keras.losses.BinaryCrossentropy(from_logits=True)]
    metrics += [[losses_lib.F1Score(from_logits=True, is_sequence=True)]]
    loss_weights += [1.0]
  keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=loss,
                      loss_weights=loss_weights,
                      metrics=metrics)
  if os.path.exists(model_path):
    print('Loading model weights from: %s' % model_path)
    keras_model.load_weights(filepath=model_path)
  else:
    print('No model weights found in %s. Starting new training...' % model_path)
  print(keras_model.summary())
  # Create batch data generator
  batch_generator = BatchGenerator(strongly_labeled_data_path, weakly_labeled_data_path, unlabeled_data_path,
                                   real_strongly_labeled_data_path=real_strongly_labeled_data_path,
                                   batch_size=opt.batch_size,
                                   normalization=normalization)
  # Create batch data generator for validation data
  val_batch_generator = BatchGenerator(strongly_labeled_test_data_path, weakly_labeled_test_data_path, unlabeled_test_data_path,
                                       real_strongly_labeled_data_path=real_strongly_labeled_test_data_path,
                                       batch_size=opt.batch_size,
                                       normalization=normalization)
  # Define callbacks
  logs_path = os.path.join(output_dir, 'logs', '{}'.format(model_name))
  os.makedirs(logs_path, exist_ok=True)
  callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                  monitor='val_out_f1_score',
                                                  mode='max',
                                                  save_weights_only=True,
                                                  save_best_only=True,
                                                  verbose=1),
               tf.keras.callbacks.TensorBoard(log_dir=logs_path)]
  # Fit model
  history = keras_model.fit(x=batch_generator,
                            validation_data=val_batch_generator,
                            epochs=training_epochs,
                            callbacks=callbacks,
                            verbose=2,
                            initial_epoch=initial_epoch)
  print('Finished training.')
  return history
