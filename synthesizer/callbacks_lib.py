import glob
import os
import math
import random
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import synthesizer.model_lib as model_lib
import synthesizer.hparams as opt
import synthesizer.audio_lib as audio_lib
import synthesizer.io_lib as io_lib
from librosa import display
from tqdm import tqdm


class VAECallback(tf.keras.callbacks.Callback):
  def __init__(self, data_path, model_path, save_path, normalization,
               infer_freq_epochs=50, num_iterations=10, num_perturbations=5):
    super(VAECallback, self).__init__()
    self.data = glob.glob(os.path.join(data_path, '*.npz'))
    self.model_path = model_path
    self.save_path = save_path
    self.normalization = normalization
    self.infer_freq_epochs = infer_freq_epochs
    self.num_iterations = num_iterations
    self.num_perturbations = num_perturbations
    if opt.features == 'mel' or opt.features == 'gammatone':
      num_features = opt.num_mel_bins
    else:
      num_features = opt.n_fft // 2 + 1
    self.vae = model_lib.Model(state_size=opt.state_size,
                               num_latents=opt.num_latents,
                               num_features=num_features)
    self.encoder_inference_model = self.vae.create_encoder_inference_model()
    self.decoder_inference_model = self.vae.create_decoder_inference_model()

  def update_inference_models(self):
    self.encoder_inference_model.load_weights(self.model_path, by_name=True)
    self.decoder_inference_model.load_weights(self.model_path, by_name=True)

  @staticmethod
  def sample_from_mvn(mvn_params, num_perturbations):
    scale_tril = tfp.bijectors.FillScaleTriL()(mvn_params[:, opt.num_latents:])
    mvn = tfp.distributions.MultivariateNormalTriL(loc=mvn_params[:, :opt.num_latents], scale_tril=scale_tril)
    all_latents = mvn.sample(num_perturbations)
    return all_latents

  def generate_model_outputs(self, sampled_data, dst_dir):
    all_mvn_params = []
    for i in tqdm(range(self.num_iterations)):
      with np.load(sampled_data[i]) as loaded_data:
        audio_sample = loaded_data['features']
        audio_sample = self.normalization.normalize(audio_sample)
      self.write_audio(audio_sample, filename=os.path.join(dst_dir, 'inputs_%d.wav' % i))
      mvn_params = self.encoder_inference_model.predict_on_batch(np.expand_dims(audio_sample, axis=0))
      mvn_params = np.squeeze(mvn_params, axis=0)
      all_mvn_params.append(mvn_params)
      all_sampled_latents = self.sample_from_mvn(mvn_params, num_perturbations=self.num_perturbations)
      all_outputs = []
      for idx, sampled_latents in enumerate(all_sampled_latents):
        outputs = self.decoder_inference_model.predict_on_batch(np.expand_dims(sampled_latents, axis=0))
        outputs = np.squeeze(outputs, axis=0)
        all_outputs.append(outputs)
        self.write_audio(outputs, filename=os.path.join(dst_dir, 'outputs_%d.%d.wav' % (i, idx)))
      self.plot([audio_sample, all_outputs], filename=os.path.join(dst_dir, 'results_%d.png' % i))
    if opt.num_latents == 2:
      self.plot_latent_space(all_mvn_params, filename=os.path.join(dst_dir, 'latent_space.png'))

  def on_epoch_end(self, epoch, logs=None):
    if (self.infer_freq_epochs != 0) and ((epoch + 1) % self.infer_freq_epochs == 0):
      self.update_inference_models()
      sampled_data = random.sample(self.data, k=self.num_iterations)
      dst_dir = os.path.join(self.save_path, 'epoch_%s' % epoch)
      os.makedirs(dst_dir, exist_ok=True)
      self.generate_model_outputs(sampled_data, dst_dir)

  def plot(self, inputs, filename):
    audio_sample, all_outputs = inputs
    frame_step = math.ceil(opt.sample_rate * (opt.frame_step_ms / 1000))
    num_plot_rows = 1 + len(all_outputs)
    plt.figure(figsize=(20, 5 * num_plot_rows))
    plt.subplot(num_plot_rows, 1, 1)
    plt.title('Inputs')
    display.specshow(self.normalization.denormalize(audio_sample).transpose(),
                     x_axis='time', y_axis='mel' if opt.features == 'mel' else 'log', sr=opt.sample_rate,
                     fmin=opt.fmin_hz, fmax=opt.fmax_hz, hop_length=frame_step, cmap='magma')
    idx = 2
    for outputs in all_outputs:
      plt.subplot(num_plot_rows, 1, idx)
      plt.title('Outputs')
      display.specshow(self.normalization.denormalize(outputs).transpose(),
                       x_axis='time', y_axis='mel' if opt.features == 'mel' else 'log', sr=opt.sample_rate,
                       fmin=opt.fmin_hz, fmax=opt.fmax_hz, hop_length=frame_step, cmap='magma')
      idx += 1
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

  @staticmethod
  def plot_latent_space(all_mvn_params, filename):
    x = []
    y = []
    for mvn_params in all_mvn_params:
      mu = mvn_params[:opt.num_latents]
      x.append(mu[0])
      y.append(mu[1])
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, 10, 'b', alpha=0.25)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

  def write_audio(self, inputs, filename):
    denormalized_inputs = self.normalization.denormalize(inputs)
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
