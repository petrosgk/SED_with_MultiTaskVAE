import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import synthesizer.io_lib as io_lib
import synthesizer.audio_lib as audio_lib
import synthesizer.utils_lib as utils_lib
import synthesizer.model_lib as model_lib
import synthesizer.hparams as opt
from tqdm import tqdm
from librosa import display


def infer(audio_files, data_path, output_dir, model_path, num_perturbations=5):
  if not os.path.exists(model_path):
    raise RuntimeError('Model weights not found in %s.' % model_path)
  os.makedirs(output_dir, exist_ok=True)
  normalization_class_path = os.path.join(data_path, 'normalization.pickle')
  with open(normalization_class_path, 'rb') as f:
    normalization = pickle.load(f)
  if opt.features == 'mel' or opt.features == 'gammatone':
    num_features = opt.num_mel_bins
  else:
    num_features = opt.n_fft // 2 + 1
  vae = model_lib.Model(state_size=opt.state_size,
                        num_latents=opt.num_latents,
                        num_features=num_features)
  vae_encoder_model = vae.create_encoder_inference_model()
  vae_decoder_model = vae.create_decoder_inference_model()
  print(vae_encoder_model.summary())
  print(vae_decoder_model.summary())
  print('Loading model weights from: %s' % model_path)
  vae_encoder_model.load_weights(model_path, by_name=True)
  vae_decoder_model.load_weights(model_path, by_name=True)
  for audio_file_idx, audio_file in tqdm(enumerate(audio_files)):
    audio_data = io_lib.load_audio_data(audio_file)
    features = utils_lib.extract_features_from_audio(audio_data)
    features = normalization.normalize(features)
    filename = os.path.splitext(os.path.basename(audio_file))[0]
    dst_dir = os.path.join(output_dir, filename)
    os.makedirs(dst_dir, exist_ok=True)
    write_audio(features,
                filename=os.path.join(dst_dir, 'inputs.wav'),
                normalization=normalization)
    mvn_params = vae_encoder_model.predict_on_batch(np.expand_dims(features, axis=0))
    mvn_params = np.squeeze(mvn_params, axis=0)
    all_sampled_latents = sample_from_mvn(mvn_params, num_perturbations=num_perturbations)
    all_outputs = []
    for latent_idx, sampled_latents in enumerate(all_sampled_latents):
      outputs = vae_decoder_model.predict_on_batch(np.expand_dims(sampled_latents, axis=0))
      outputs = np.squeeze(outputs, axis=0)
      all_outputs.append(outputs)
      write_audio(outputs,
                  filename=os.path.join(dst_dir, 'outputs_%d.wav' % latent_idx),
                  normalization=normalization)
    plot([features, all_outputs], filename=os.path.join(output_dir, 'results.png'), normalization=normalization)


def sample_from_mvn(mvn_params, num_perturbations):
  scale_tril = tfp.bijectors.FillScaleTriL()(mvn_params[:, opt.num_latents:])
  mvn = tfp.distributions.MultivariateNormalTriL(loc=mvn_params[:, :opt.num_latents], scale_tril=scale_tril)
  all_latents = mvn.sample(num_perturbations)
  return all_latents


def write_audio(inputs, filename, normalization):
  denormalized_inputs = normalization.denormalize(inputs)
  if opt.features == 'mel':
    audio_data = audio_lib.extract_audio_from_mel_features(denormalized_inputs)
  elif opt.features == 'stft':
    audio_data = audio_lib.extract_audio_from_stft_features(denormalized_inputs)
  else:
    audio_data = None
  if audio_data is None:
    print('Skipping audio file %s' % filename)
    return
  io_lib.write_audio_data(audio_data, filename)


def plot(inputs, filename, normalization):
  audio_sample, all_outputs = inputs
  frame_step = math.ceil(opt.sample_rate * (opt.frame_step_ms / 1000))
  num_plot_rows = 1 + len(all_outputs)
  plt.figure(figsize=(20, 5 * num_plot_rows))
  plt.subplot(num_plot_rows, 1, 1)
  plt.title('Inputs')
  display.specshow(normalization.denormalize(audio_sample).transpose(),
                   x_axis='time', y_axis='mel' if opt.features == 'mel' else 'log', sr=opt.sample_rate,
                   fmin=opt.fmin_hz, fmax=opt.fmax_hz, hop_length=frame_step, cmap='magma')
  idx = 2
  for outputs in all_outputs:
    plt.subplot(num_plot_rows, 1, idx)
    plt.title('Outputs')
    display.specshow(normalization.denormalize(outputs).transpose(),
                     x_axis='time', y_axis='mel' if opt.features == 'mel' else 'log', sr=opt.sample_rate,
                     fmin=opt.fmin_hz, fmax=opt.fmax_hz, hop_length=frame_step, cmap='magma')
    idx += 1
  plt.tight_layout()
  plt.savefig(filename)
  plt.close()