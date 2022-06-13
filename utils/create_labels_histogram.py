import os
import glob
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm


labels = ['Speech',
          'Dog',
          'Cat',
          'Alarm_bell_ringing',
          'Dishes',
          'Frying',
          'Blender',
          'Running_water',
          'Vacuum_cleaner',
          'Electric_shaver_toothbrush']


def read_transcription(transcription):
  with open(transcription, 'r') as f:
    lines = f.readlines()
  metadata = []
  for line in lines:
    onset, offset, label = line.split('\t')
    label = label.rstrip('\n')
    duration = float(offset) - float(onset)
    metadata.append(
      {'duration': duration, 'label': label}
    )
  return metadata


def build_histogram_data(transcriptions):
  duration_per_label = {}
  for label in labels:
    duration_per_label[label] = 0.0
  for transcription in tqdm(transcriptions):
    metadata = read_transcription(transcription)
    for entry in metadata:
      duration = entry['duration']
      label = entry['label']
      duration_per_label[label] += duration
  # Seconds -> minutes
  for label, duration in duration_per_label.items():
    duration_per_label[label] /= 60
  return duration_per_label


def plot_histogram(duration_per_label):
  fig, ax = plt.subplots()
  ax.barh(labels, duration_per_label.values(), 0.5)
  ax.set_ylabel('Label')
  ax.set_xlabel('Duration (minutes)')
  fig.tight_layout()
  plt.show()


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_transcriptions', required=True, type=str,
                      help='Path to transcriptions.')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  transcriptions = glob.glob(os.path.join(args.path_to_transcriptions, '*.txt'))
  duration_per_label = build_histogram_data(transcriptions)
  plot_histogram(duration_per_label)
