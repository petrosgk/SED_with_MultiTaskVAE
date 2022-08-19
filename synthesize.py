import os
import glob
import argparse
import synthesizer.infer_lib as infer_lib


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_audio_files', type=str,
                      help='Path to audio file(s). '
                           'Can be a path to a single audio file or to a directory containing multiple audio files.')
  parser.add_argument('--path_to_normalization_class', required=True, type=str,
                      help='Path to normalization class .pickle file.')
  parser.add_argument('--path_to_model', required=True, type=str,
                      help='Path to trained model checkpoint.')
  parser.add_argument('--output_dir', required=True, type=str,
                      help='Directory to store outputs.')
  parser.add_argument('--num_perturbations', type=int, default=3,
                      help='Number of variations to generated for each input audio file.')
  args = parser.parse_args()
  return args


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  args = parse_args()
  if os.path.isdir(args.path_to_audio_files):
    audio_files = glob.glob(os.path.join(args.path_to_audio_files, '*.wav'))
    assert audio_files, 'No audio files found in "%s"' % args.path_to_audio_files
  else:
    audio_files = [args.path_to_audio_files]
  infer_lib.infer(audio_files=audio_files,
                  normalization_class=args.path_to_normalization_class,
                  output_dir=args.output_dir,
                  model_path=args.path_to_model,
                  num_perturbations=args.num_perturbations)
