import math
import librosa
import event_detector.hparams as opt


def load_audio_data(audio_file):
  # Load audio file
  audio_data, _ = librosa.core.load(path=audio_file, sr=opt.sample_rate)
  return audio_data


def extract_utterances_from_probs(probs, labels, threshold):
  frame_step = math.ceil(opt.sample_rate * (opt.frame_step_ms / 1000))
  utterances = []
  for label_idx in range(len(labels)):
    index = 0
    while index < len(probs):
      if probs[index, label_idx] >= threshold:
        start = index
        while (index < len(probs)) and (probs[index, label_idx] >= threshold):
          index += 1
        end = index - 1
        start_timestamp = (start * frame_step) / opt.sample_rate
        end_timestamp = (end * frame_step) / opt.sample_rate
        utterances.append({'start': start_timestamp,
                           'end': end_timestamp,
                           'label': labels[label_idx]})
      index += 1
  sorted_utterances = sorted(utterances, key=lambda x: x['start'])
  return sorted_utterances


def extract_transcription(filename, probs, labels, threshold):
  # Extract utterances from class probabilities per frame
  utterances = extract_utterances_from_probs(probs, labels=labels, threshold=threshold)
  # Convert utterances to a transcription format
  transcription_lines = []
  for utterance in utterances:
    transcription_line = '%.3f\t%.3f\t%s\n' % (utterance['start'], utterance['end'], utterance['label'])
    transcription_lines.append(transcription_line)
  with open(filename, 'w') as f:
    # Write transcription lines to transcription file
    f.writelines(transcription_lines)
  return transcription_lines


def post_process(inputs, num_labels, min_gap_frames, padding_frames, threshold):
  if (min_gap_frames == 0) and (padding_frames == 0):
    return inputs

  # Process predictions for each label
  for label_idx in range(num_labels):

    # Fill label to short valley
    offset = False
    offset_point = None
    for i in range(inputs.shape[0]):
      if i < inputs.shape[0] - 1:
        # offset detection
        if (inputs[i, label_idx] >= threshold) and (inputs[i + 1, label_idx] < threshold):
          offset = True
          offset_point = i
        # offset -> onset detection
        if (inputs[i, label_idx] < threshold) and (inputs[i + 1, label_idx] >= threshold) and offset:
          if i - offset_point < min_gap_frames:
            # Fill label to valley
            inputs[offset_point:i + 1, label_idx] = 1
            offset = False

    # Remove impulse-like detection
    onset = False
    onset_point = None
    for i in range(inputs.shape[0]):
      if i < inputs.shape[0] - 1:
        # onset detection
        if (inputs[i, label_idx] < threshold) and (inputs[i + 1, label_idx] >= threshold):
          onset = True
          onset_point = i
        # onset -> offset detection
        if (inputs[i, label_idx] >= threshold) and (inputs[i + 1, label_idx] < threshold) and onset:
          if i - onset_point < min_gap_frames:
            # Fill zeros to hill
            inputs[onset_point:i + 1, label_idx] = 0
            onset = False

    # Hang before & over
    onset = False
    # Special case where predictions begin with a non-unvoiced label
    if inputs[0, label_idx] >= threshold:
      onset = True
    for i in range(inputs.shape[0]):
      if i < inputs.shape[0] - 1:
        # onset detection
        if (inputs[i, label_idx] < threshold) and (inputs[i + 1, label_idx] >= threshold):
          onset = True
          if i - padding_frames < 0:
            inputs[0:i + 1, label_idx] = 1
          else:
            inputs[i - padding_frames:i + 1, label_idx] = 1
        # onset -> offset detection
        if (inputs[i, label_idx] >= threshold) and (inputs[i + 1, label_idx] < threshold) and onset:
          onset = False
          if i + padding_frames > inputs.shape[0]:
            inputs[i:, label_idx] = 1
          else:
            inputs[i:i + padding_frames, label_idx] = 1

  return inputs