import math
import numpy as np
import librosa
import pyloudnorm
import synthesizer.hparams as opt
from gammatone import gtgram
import warnings


def remove_dc(inputs):
  return inputs - np.mean(inputs)


def extract_stft_features_from_audio(inputs):
  inputs = remove_dc(inputs)
  if opt.loudness_normalize:
    inputs = loudness_normalize_audio(inputs)
  frame_step = round(opt.sample_rate * (opt.frame_step_ms / 1000))
  try:
    stft = librosa.core.stft(inputs,
                             n_fft=opt.n_fft,
                             pad_mode='constant',
                             hop_length=frame_step)
  except librosa.ParameterError as e:
    print(e)
    return None
  log_amplitude = librosa.core.amplitude_to_db(np.abs(stft), top_db=opt.top_db).transpose()
  return log_amplitude


def extract_mel_features_from_audio(inputs):
  inputs = remove_dc(inputs)
  if opt.loudness_normalize:
    inputs = loudness_normalize_audio(inputs)
  frame_step = round(opt.sample_rate * (opt.frame_step_ms / 1000))
  try:
    mel = librosa.feature.melspectrogram(inputs,
                                         sr=opt.sample_rate,
                                         n_fft=opt.n_fft,
                                         pad_mode='constant',
                                         hop_length=frame_step,
                                         n_mels=opt.num_mel_bins,
                                         fmin=opt.fmin_hz,
                                         fmax=opt.fmax_hz)
  except librosa.ParameterError as e:
    print(e)
    return None
  log_mel = librosa.core.amplitude_to_db(np.abs(mel), top_db=opt.top_db).transpose()
  return log_mel


def extract_gammatone_features_from_audio(inputs):
  inputs = remove_dc(inputs)
  if opt.loudness_normalize:
    inputs = loudness_normalize_audio(inputs)
  window_time = opt.n_fft / opt.sample_rate
  hop_time = opt.frame_step_ms / 1000
  gammatonegram = gtgram.gtgram(inputs,
                                fs=opt.sample_rate,
                                window_time=window_time,
                                hop_time=hop_time,
                                channels=opt.num_mel_bins,
                                f_min=opt.fmin_hz)
  log_amplitude = librosa.core.amplitude_to_db(np.abs(gammatonegram), top_db=opt.top_db).transpose()
  return log_amplitude


def extract_audio_from_stft_features(inputs):
  amplitude = librosa.core.db_to_amplitude(inputs).transpose()
  frame_step = math.ceil(opt.sample_rate * (opt.frame_step_ms / 1000))
  try:
    audio_data = librosa.griffinlim(amplitude,
                                    hop_length=frame_step,
                                    pad_mode='constant')
  except librosa.util.exceptions.ParameterError as e:
    print(e)
    return None
  if not np.all(np.isfinite(inputs)):
    return None
  if opt.loudness_normalize:
    audio_data = loudness_normalize_audio(audio_data)
  return audio_data


def loudness_normalize_audio(inputs, block_size=0.4):
  warnings.simplefilter('ignore')
  audio_duration = len(inputs)
  min_audio_duration = math.ceil(block_size * opt.sample_rate)
  if audio_duration < min_audio_duration:
    inputs = np.pad(inputs, pad_width=(0, min_audio_duration - audio_duration))
  # measure the loudness first
  meter = pyloudnorm.Meter(opt.sample_rate, block_size=block_size)  # create BS.1770 meter
  loudness = meter.integrated_loudness(inputs)
  # loudness normalize audio to -3 dB LUFS
  loudness_normalized_audio = pyloudnorm.normalize.loudness(inputs, loudness, -3.0)
  return loudness_normalized_audio
