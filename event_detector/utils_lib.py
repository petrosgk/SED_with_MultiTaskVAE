import os
import numpy as np
import tensorflow as tf
import event_detector.io_lib as io_lib
import event_detector.audio_lib as audio_lib
import event_detector.hparams as opt
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


def mask_from_strong_label_file(label_file, length, labels):
  mask = np.zeros(shape=(length, len(labels)), dtype=np.float32)
  with open(label_file, 'r') as f:
    lines = f.readlines()
  for line in lines:
    start, end, label = line.split(sep='\t')
    label = label.rstrip('\n')
    start = round(float(start) / (opt.frame_step_ms / 1000))
    end = round(float(end) / (opt.frame_step_ms / 1000))
    label_idx = labels.index(label)
    mask[start:end, label_idx] = 1
  return mask


def mask_from_weak_label_file(label_file, labels):
  mask = np.zeros(shape=len(labels), dtype=np.float32)
  with open(label_file, 'r') as f:
    lines = f.readlines()
  for line in lines:
    label = line.rstrip('\n')
    label_idx = labels.index(label)
    mask[label_idx] = 1
  return mask


def extract_features_from_audio(audio_data):
  if opt.mel_features:
    features = audio_lib.extract_mel_features_from_audio(audio_data)
  elif opt.gammatone_features:
    features = audio_lib.extract_gammatone_features_from_audio(audio_data)
  else:
    features = audio_lib.extract_stft_features_from_audio(audio_data)
  return features


def data_extractor(audio_files, save_path, normalization=None, weak_labels=False, unlabeled=False):
  audio_file_idx = 0
  for audio_file in tqdm(audio_files):
    if not os.path.exists(audio_file):
      raise FileNotFoundError('%s not found.' % audio_file)
    label_file = None
    if not unlabeled:
      label_file = os.path.splitext(audio_file)[0] + '.txt'
      if not os.path.exists(label_file):
        print('\nSkipping audio file %s' % audio_file)
        continue
    audio_data = io_lib.load_audio_data(audio_file)
    features = extract_features_from_audio(audio_data)
    if features is None:
      print('\nSkipping audio file %s' % audio_file)
      continue
    if normalization is not None:
      normalization.update_statistics(features)
    mask = None
    if not unlabeled:
      if weak_labels:
        mask = mask_from_weak_label_file(label_file, labels=opt.labels)
      else:
        mask = mask_from_strong_label_file(label_file, length=len(features), labels=opt.labels)
    dst = os.path.join(save_path, 'data' + '.' + str(audio_file_idx) + '.npz')
    with open(dst, 'wb') as f:
      if mask is not None:
        np.savez_compressed(f, features=features, mask=mask)
      else:
        np.savez_compressed(f, features=features)
    audio_file_idx += 1
