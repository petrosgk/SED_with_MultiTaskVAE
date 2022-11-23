import glob
import os
import argparse
import synthesizer.train_lib as train_lib
import synthesizer.hparams as opt


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_data', required=True, type=str,
                      help='Path to training data.')
  parser.add_argument('--output_dir', required=True, type=str,
                      help='Directory to place model data.')
  parser.add_argument('--model_name', required=True, type=str,
                      help='Model name.')
  parser.add_argument('--initial_epoch', type=int, default=0,
                      help='Initial epoch. Useful when resuming training.')
  parser.add_argument('--initial_lr', default=None, type=float,
                      help='Initial learning. Useful when resuming training.')
  args = parser.parse_args()
  return args


def filter_audio_files(audio_files):
  filtered_audio_files = []
  label_per_audio_file = []
  for audio_file in audio_files:
    for label in opt.labels:
      if label in audio_file:
        filtered_audio_files.append(audio_file)
        label_per_audio_file.append(label)
  return filtered_audio_files, label_per_audio_file


if __name__ == '__main__':
  args = parse_args()
  audio_files = glob.glob(
    os.path.join(
      os.path.join(args.path_to_data, "*"),
      '*.wav'
    )
  )
  filtered_audio_files, label_per_audio_file = filter_audio_files(audio_files)
  train_lib.train(audio_files=filtered_audio_files,
                  output_dir=args.output_dir,
                  label_per_audio_file=label_per_audio_file,
                  model_name=args.model_name,
                  initial_epoch=args.initial_epoch,
                  initial_lr=args.initial_lr)
