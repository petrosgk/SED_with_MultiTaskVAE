import glob
import os
import argparse
import numpy as np
import multiprocessing as mp
import event_detector.hparams as opt


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
  parser.add_argument('--post_process', action='store_true',
                      help='Apply post-processing to model predictions.')
  parser.add_argument('--save_raw_outputs', action='store_true',
                      help='Save raw model outputs (probabilities) as .csv and .npy files.')
  parser.add_argument('--threshold', type=float,
                      help='Single threshold to use for evaluating event detector. '
                           'If not specified a range of thresholds will be used.')
  parser.add_argument('--num_workers', type=int, default=10,
                      help='Number of processes to use.')
  args = parser.parse_args()
  return args


def run_sed(reference_transcripts, save_path, print_results):
  my_path = os.path.abspath(os.path.dirname(__file__))
  with open(os.path.join(save_path, 'sed_file_list.txt'), 'w') as f:
    for reference_transcript in reference_transcripts:
      predicted_transcript = os.path.join(
        save_path, os.path.basename(os.path.splitext(reference_transcript)[0])
      ) + '.txt'
      if not os.path.exists(predicted_transcript):
        continue
      f.write(reference_transcript + '\t' + predicted_transcript + '\n')
  if print_results:
    command = 'python {} {}'.format(
      os.path.join(my_path, 'evaluators', 'sound_event_eval.py'),
      os.path.join(save_path, 'sed_file_list.txt')
    )
  else:
    command = 'python {} {} -o {}'.format(
      os.path.join(my_path, 'evaluators', 'sound_event_eval.py'),
      os.path.join(save_path, 'sed_file_list.txt'),
      os.path.join(save_path, 'results.yaml')
    )
  print(command)
  os.system(command)


def worker(args, audio_files, reference_transcripts, thresholds):
  import event_detector.infer_lib as infer_lib
  for threshold in thresholds:
    print(f'Evaluating with threshold = {threshold}')
    save_path = os.path.join(args.output_dir, 'threshold_' + str(round(threshold, ndigits=2)))
    os.makedirs(save_path, exist_ok=True)
    infer_lib.infer(audio_files=audio_files,
                    data_path=args.path_to_data,
                    save_path=save_path,
                    model_path=args.path_to_model,
                    save_raw_outputs=args.save_raw_outputs,
                    post_process=args.post_process,
                    threshold=threshold)
    print('Running evaluation...')
    print_results = False
    if threshold == opt.threshold:
      print_results = True
    run_sed(reference_transcripts, save_path, print_results=print_results)


if __name__ == '__main__':
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  args = parse_args()
  audio_files = glob.glob(os.path.join(args.path_to_evaluation_data, '*.wav'))
  reference_transcripts = glob.glob(os.path.join(args.path_to_evaluation_data, '*.txt'))
  # Evaluate detector on 50 operating points linearly distributed from 0.01 to 0.99
  if not args.threshold:
    thresholds = np.arange(0.01, 1.0, 0.02)
  else:
    thresholds = [args.threshold]
  thresholds_per_worker = len(thresholds) // args.num_workers
  workers = []
  for worker_id in range(args.num_workers):
    if worker_id < (args.num_workers - 1):
      worker_thresholds = thresholds[thresholds_per_worker * worker_id:thresholds_per_worker * (worker_id + 1)]
    else:
      worker_thresholds = thresholds[thresholds_per_worker * worker_id:]
    p = mp.Process(target=worker, args=(args, audio_files, reference_transcripts, worker_thresholds))
    p.start()
    workers.append(p)
  for worker in workers:
    worker.join()
  print('Done.')