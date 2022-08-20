import numpy as np
import os
import pickle
import event_detector.model_lib as model_lib
import event_detector.hparams as opt
import event_detector.io_lib as io_lib
import event_detector.audio_lib as audio_lib
from tqdm import tqdm
from scipy.signal import medfilt


def infer(audio_files, normalization_class, output_dir, model_path, thresholds, save_raw_probs=False):
  if not os.path.exists(model_path):
    raise RuntimeError('Model weights not found in %s.' % model_path)
  os.makedirs(output_dir, exist_ok=True)
  print('Loading normalization class from: %s' % normalization_class)
  with open(normalization_class, 'rb') as f:
    normalization = pickle.load(f)
  if opt.features == 'mel' or opt.features == 'gammatone':
    num_features = opt.num_mel_bins
  else:
    num_features = opt.n_fft // 2 + 1
  ead = model_lib.Model(state_size=opt.state_size,
                        num_latents=opt.num_latents,
                        variational_encoder=opt.variational_encoder,
                        num_features=num_features,
                        num_labels=len(opt.labels))
  model = ead.create_inference_model()
  print(model.summary())
  print('Loading model weights from: %s' % model_path)
  model.load_weights(model_path, by_name=True)
  metadata_tsv_file = open(os.path.join(output_dir, 'metadata.tsv'), 'w')
  metadata_tsv_file.write('filename' + '\t' + 'duration' + '\n')
  probs_per_audio_file = []
  for audio_file in tqdm(audio_files):
    if not os.path.exists(audio_file):
      raise FileNotFoundError('%s not found.' % audio_file)
    # Prepare audio data for inference
    audio_data = io_lib.load_audio_data(audio_file)
    duration = round(len(audio_data) / opt.sample_rate, ndigits=1)
    metadata_tsv_file.write(os.path.basename(audio_file) + '\t' + str(duration) + '\n')
    if opt.features == 'mel':
      inputs = audio_lib.extract_mel_features_from_audio(audio_data)
    elif opt.features == 'gammatone':
      inputs = audio_lib.extract_gammatone_features_from_audio(audio_data)
    else:
      inputs = audio_lib.extract_stft_features_from_audio(audio_data)
    inputs = normalization.normalize(inputs)
    # Infer from audio features
    # If strong labels, model outputs are event class probabilities per audio frame
    # If weak labels, model outputs are event class probabilities for the audio file
    probs = model.predict_on_batch(np.expand_dims(inputs, axis=0))
    probs = np.squeeze(probs, axis=0)
    probs_per_audio_file.append(
      {'filename': os.path.basename(audio_file),
       'probs': probs}
    )
    if save_raw_probs:
      raw_probs_save_dir = os.path.join(output_dir, 'raw_probs')
      os.makedirs(raw_probs_save_dir, exist_ok=True)
      fname = os.path.join(raw_probs_save_dir, os.path.splitext(os.path.basename(audio_file))[0])
      np.savetxt(fname=fname + '.csv', X=probs, delimiter=',')
      np.save(file=fname + '.npy', arr=probs)
  metadata_tsv_file.close()
  for threshold in thresholds:
    print(f'Threshold = {round(threshold, ndigits=2)}')
    save_path = os.path.join(output_dir, 'threshold_' + str(round(threshold, ndigits=2)))
    os.makedirs(save_path, exist_ok=True)
    outputs_tsv_file = open(os.path.join(save_path, 'outputs.tsv'), 'w')
    outputs_tsv_file.write('filename' + '\t' + 'onset' + '\t' + 'offset' + '\t' + 'event_label' + '\n')
    for probs in tqdm(probs_per_audio_file):
      filename = probs['filename']
      raw_probs = probs['probs']
      # Apply median filtering
      processed_raw_probs = medfilt(raw_probs, kernel_size=(19, 1))
      # Apply thresholding and convert model probability outputs to transcription
      transcription_filename = os.path.join(save_path, os.path.splitext(filename)[0] + '.txt')
      transcription_lines = io_lib.extract_transcription(transcription_filename, probs=processed_raw_probs, labels=opt.labels, threshold=threshold)
      for transcription_line in transcription_lines:
        outputs_tsv_file.write(filename + '\t' + transcription_line)
    outputs_tsv_file.close()
