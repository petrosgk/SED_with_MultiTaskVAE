import glob
import os
import argparse
import event_detector.train_lib as train_lib


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_strongly_labeled_train_data', required=True, type=str,
                      help='Path to strongly labeled training data.')
  parser.add_argument('--path_to_strongly_labeled_test_data', required=True, type=str,
                      help='Path to strongly labeled test data.')
  parser.add_argument('--path_to_weakly_labeled_train_data', required=True, type=str,
                      help='Path to weakly labeled training data.')
  parser.add_argument('--path_to_unlabeled_train_data', required=True, type=str,
                      help='Path to unlabeled training data.')
  parser.add_argument('--path_to_real_strongly_labeled_train_data', type=str,
                      help='Path to real strongly labeled training data.')
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


if __name__ == '__main__':
  args = parse_args()
  strongly_labeled_train_audio_files = glob.glob(os.path.join(args.path_to_strongly_labeled_train_data, '*.wav'))
  strongly_labeled_test_audio_files = glob.glob(os.path.join(args.path_to_strongly_labeled_test_data, '*.wav'))
  weakly_labeled_train_audio_files = glob.glob(os.path.join(args.path_to_weakly_labeled_train_data, '*.wav'))
  unlabeled_train_audio_files = glob.glob(os.path.join(args.path_to_unlabeled_train_data, '*.wav'))
  real_strongly_labeled_train_audio_files = None
  if args.path_to_real_strongly_labeled_train_data:
    real_strongly_labeled_train_audio_files = glob.glob(os.path.join(args.path_to_real_strongly_labeled_train_data, '*.wav'))
  train_lib.train(strongly_labeled_audio_files=strongly_labeled_train_audio_files,
                  test_audio_files=strongly_labeled_test_audio_files,
                  weakly_labeled_audio_files=weakly_labeled_train_audio_files,
                  unlabeled_audio_files=unlabeled_train_audio_files,
                  real_strongly_labeled_audio_files=real_strongly_labeled_train_audio_files,
                  output_dir=args.output_dir,
                  model_name=args.model_name,
                  initial_epoch=args.initial_epoch,
                  initial_lr=args.initial_lr)
