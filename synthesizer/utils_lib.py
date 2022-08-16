import os
import numpy as np
import tensorflow as tf
import synthesizer.io_lib as io_lib
import synthesizer.audio_lib as audio_lib
import synthesizer.hparams as opt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


class Normalization:
  def __init__(self):
    self.scaler = StandardScaler()

  def update_statistics(self, inputs):
    self.scaler.partial_fit(inputs)

  def normalize(self, inputs):
    return self.scaler.transform(inputs)

  def denormalize(self, inputs):
    return self.scaler.inverse_transform(inputs)


def force_gpu_memory_growth():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)


def extract_features_from_audio(audio_data):
  if opt.mel_features:
    features = audio_lib.extract_mel_features_from_audio(audio_data)
  elif opt.gammatone_features:
    features = audio_lib.extract_gammatone_features_from_audio(audio_data)
  else:
    features = audio_lib.extract_stft_features_from_audio(audio_data)
  return features


def data_extractor(audio_files, save_path, normalization=None):
  features_lengths = []
  audio_file_idx = 0
  for audio_file in tqdm(audio_files):
    if not os.path.exists(audio_file):
      raise FileNotFoundError('%s not found.' % audio_file)
    audio_data = io_lib.load_audio_data(audio_file)
    features = extract_features_from_audio(audio_data)
    if features is None:
      print('\nSkipping audio file %s' % audio_file)
      continue
    features_lengths.append(len(features))
    if normalization is not None:
      normalization.update_statistics(features)
    dst = os.path.join(save_path, 'data' + '.' + str(audio_file_idx) + '.npz')
    with open(dst, 'wb') as f:
      np.savez_compressed(f, features=features)
    audio_file_idx += 1
  return min(features_lengths), max(features_lengths)
