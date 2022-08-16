import glob
import os
import argparse
import numpy as np
import event_detector.infer_lib as infer_lib


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_evaluation_data', required=True, type=str,
                      help='Path to audio files.')
  parser.add_argument('--path_to_data', required=True, type=str,
                      help='Path to data.')
  parser.add_argument('--path_to_model', required=True, type=str,
                      help='Path to trained model checkpoint.')
  parser.add_argument('--output_dir', required=True, type=str,
                      help='Directory to store predicted transcripts.')
  parser.add_argument('--save_raw_outputs', action='store_true',
                      help='Save raw model outputs (probabilities) as .csv and .npy files.')
  parser.add_argument('--debug', action='store_true',
                      help='Only use a single threshold of 0.51 for testing instead of all thresholds.')
  args = parser.parse_args()
  return args


def run_sed(reference_transcripts, save_path):
  my_path = os.path.abspath(os.path.dirname(__file__))
  with open(os.path.join(save_path, 'sed_file_list.txt'), 'w') as f:
    for reference_transcript in reference_transcripts:
      predicted_transcript = os.path.join(
        save_path, os.path.basename(os.path.splitext(reference_transcript)[0])
      ) + '.txt'
      if not os.path.exists(predicted_transcript):
        continue
      f.write(reference_transcript + '\t' + predicted_transcript + '\n')
  command = 'python {} {} -o {}'.format(
    os.path.join(my_path, 'evaluators', 'sound_event_eval.py'),
    os.path.join(save_path, 'sed_file_list.txt'),
    os.path.join(save_path, 'results.yaml')
  )
  print(command)
  os.system(command)


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  args = parse_args()
  audio_files = glob.glob(os.path.join(args.path_to_evaluation_data, '*.wav'))
  reference_transcripts = glob.glob(os.path.join(args.path_to_evaluation_data, '*.txt'))
  # Evaluate detector on 50 operating points linearly distributed from 0.01 to 0.99
  if args.debug:
    thresholds = [0.51]
  else:
    thresholds = np.arange(0.01, 1.0, 0.02)
  infer_lib.infer(audio_files=audio_files,
                  data_path=args.path_to_data,
                  output_dir=args.output_dir,
                  model_path=args.path_to_model,
                  thresholds=thresholds,
                  save_raw_probs=args.save_raw_outputs)
  print('Running evaluation...')
  run_sed(reference_transcripts, os.path.join(args.output_dir, 'threshold_' + str(0.51)))
  print('Done.')
