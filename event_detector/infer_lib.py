import numpy as np
import os
import pickle
import event_detector.model_lib as model_lib
import event_detector.hparams as opt
import event_detector.io_lib as io_lib
import event_detector.audio_lib as audio_lib
from tqdm import tqdm
from scipy.signal import medfilt


def infer(audio_files, data_path, save_path, model_path, threshold, post_process=False, save_raw_outputs=False):
  if not os.path.exists(model_path):
    raise RuntimeError('Model weights not found in %s.' % model_path)
  normalization_class_path = os.path.join(data_path, 'normalization.pickle')
  with open(normalization_class_path, 'rb') as f:
    normalization = pickle.load(f)
  if opt.mel_features or opt.gammatone_features:
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
  outputs_tsv_file = open(os.path.join(save_path, 'outputs.tsv'), 'w')
  outputs_tsv_file.write('filename' + '\t' + 'onset' + '\t' + 'offset' + '\t' + 'event_label' + '\n')
  metadata_tsv_file = open(os.path.join(save_path, 'metadata.tsv'), 'w')
  metadata_tsv_file.write('filename' + '\t' + 'duration' + '\n')
  for audio_file in tqdm(audio_files):
    if not os.path.exists(audio_file):
      raise FileNotFoundError('%s not found.' % audio_file)
    # Prepare audio data for inference
    audio_data = io_lib.load_audio_data(audio_file)
    duration = round(len(audio_data) / opt.sample_rate, ndigits=1)
    metadata_tsv_file.write(os.path.basename(audio_file) + '\t' + str(duration) + '\n')
    if opt.mel_features:
      inputs = audio_lib.extract_mel_features_from_audio(audio_data)
    elif opt.gammatone_features:
      inputs = audio_lib.extract_gammatone_features_from_audio(audio_data)
    else:
      inputs = audio_lib.extract_stft_features_from_audio(audio_data)
    inputs = normalization.normalize(inputs)
    # Infer from audio features
    # If strong labels, model outputs are event class probabilities per audio frame
    # If weak labels, model outputs are event class probabilities for the audio file
    outputs = model.predict_on_batch(np.expand_dims(inputs, axis=0))
    outputs = np.squeeze(outputs, axis=0)
    if save_raw_outputs:
      fname = os.path.join(save_path, os.path.splitext(os.path.basename(audio_file))[0])
      np.savetxt(fname=fname + '.csv', X=outputs, delimiter=',')
      np.save(file=fname + '.npy', arr=outputs)
    if post_process:
      outputs = medfilt(outputs, kernel_size=(19, 1))
    # Convert model outputs to a transcription
    filename = os.path.join(save_path, os.path.splitext(os.path.basename(audio_file))[0] + '.txt')
    transcription_lines = io_lib.extract_transcription(filename, probs=outputs, labels=opt.labels, threshold=threshold)
    for transcription_line in transcription_lines:
      outputs_tsv_file.write(os.path.basename(audio_file) + '\t' + transcription_line)
  outputs_tsv_file.close()
  metadata_tsv_file.close()
