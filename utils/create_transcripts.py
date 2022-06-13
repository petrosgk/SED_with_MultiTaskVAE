import argparse
import os


def read_metadata_file(metadata_file):
  with open(metadata_file, 'r') as f:
    lines = f.readlines()
  lines = lines[1:]  # Skip 1st line
  return lines


def read_strong_label_metadata(metadata_file):
  metadata = []
  lines = read_metadata_file(metadata_file)
  for line in lines:
    filename, onset, offset, event_label = line.split('\t')
    filename = filename.strip()
    onset = onset.strip()
    offset = offset.strip()
    event_label = event_label.strip()
    event_label = event_label.rstrip('\n')
    if not event_label:
      print('Skipping file %s' % filename)
      continue
    metadata.append(
      {'filename': filename, 'onset': onset, 'offset': offset, 'label': event_label}
    )
  return metadata


def read_weak_label_metadata(metadata_file):
  metadata = []
  lines = read_metadata_file(metadata_file)
  for line in lines:
    filename, event_labels = line.split('\t')
    filename = filename.strip()
    event_labels = event_labels.strip()
    event_labels = event_labels.rstrip('\n')
    if not event_labels:
      print('Skipping file %s' % filename)
      continue
    event_labels = event_labels.split(',')
    metadata.append(
      {'filename': filename, 'labels': event_labels}
    )
  return metadata


def read_ss_metadata(metadata_file):
  metadata = []
  lines = read_metadata_file(metadata_file)
  for line in lines:
    filename, onset, offset, event_label, isolated_background, isolated_event  = line.split('\t')
    filename = filename.strip()
    onset = onset.strip()
    offset = offset.strip()
    event_label = event_label.strip()
    isolated_background = isolated_background.strip()
    isolated_event = isolated_event.strip()
    metadata.append(
      {'filename': filename, 'onset': onset, 'offset': offset, 'label': event_label,
       'isolated_background': isolated_background, 'isolated_event': isolated_event}
    )
  return metadata



def create_strong_label_transcription_files(metadata, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  index = 0
  while index < len(metadata):
    filename = metadata[index]['filename']
    txt_file_path = os.path.join(
      output_dir,
      os.path.basename(os.path.splitext(filename)[0]) + '.txt'
    )
    with open(txt_file_path, 'w') as f:
      while True:
        label = metadata[index]['label']
        f.write(
          metadata[index]['onset'] + '\t' + metadata[index]['offset'] + '\t' + label + '\n'
        )
        index += 1
        if index == len(metadata):
          break
        if metadata[index]['filename'] != filename:
          break


def create_weak_label_transcription_files(metadata, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  index = 0
  while index < len(metadata):
    filename = metadata[index]['filename']
    txt_file_path = os.path.join(
      output_dir,
      os.path.basename(os.path.splitext(filename)[0]) + '.txt'
    )
    with open(txt_file_path, 'w') as f:
      labels = metadata[index]['labels']
      for label in labels:
        f.write(label + '\n')
    index += 1


def create_ss_transcription_files(metadata, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  index = 0
  while index < len(metadata):
    filename = metadata[index]['filename']
    txt_file_path = os.path.join(
      output_dir,
      os.path.basename(os.path.splitext(filename)[0]) + '.txt'
    )
    with open(txt_file_path, 'w') as f:
      while True:
        label = metadata[index]['label']
        f.write(
          metadata[index]['onset'] + '\t' + metadata[index]['offset'] + '\t' + label + '\t' +  metadata[index]['isolated_background'] + '\t' +  metadata[index]['isolated_event'] + '\n'
        )
        index += 1
        if index == len(metadata):
          break
        if metadata[index]['filename'] != filename:
          break


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_metadata', required=True, type=str,
                      help='Path to audio files metadata')
  parser.add_argument('--output_dir', required=True, type=str,
                      help='Directory to write output files.')
  parser.add_argument('--labels_type', choices=['strong', 'weak', 'ss'], default='strong',
                      help='Type of labels to create.')
  parser.add_argument('--num_workers', default=16, type=int,
                      help='Number of threads to use.')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  args = parse_args()
  if args.labels_type == 'strong':
    metadata = read_strong_label_metadata(args.path_to_metadata)
    create_strong_label_transcription_files(metadata, args.output_dir)
  elif args.labels_type == 'weak':
    metadata = read_weak_label_metadata(args.path_to_metadata)
    create_weak_label_transcription_files(metadata, args.output_dir)
  elif args.labels_type == 'ss':
    metadata = read_ss_metadata(args.path_to_metadata)
    create_ss_transcription_files(metadata, args.output_dir)
  else:
    raise ValueError('Unknown labels type.')