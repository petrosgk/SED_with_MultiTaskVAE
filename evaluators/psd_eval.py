import os
import glob
import argparse
import pandas as pd
import warnings
from psds_eval import PSDSEval, plot_psd_roc


def parse_args():
  parser = argparse.ArgumentParser(description='PSDS evaluator')
  parser.add_argument('--groundtruth', required=True,
                      help='Path to ground truth .tsv file.')
  parser.add_argument('--metadata', required=True,
                      help='Path to metadata .tsv file.')
  parser.add_argument('--predictions', required=True,
                      help='Path to directory containing predictions for each threshold.')
  args = parser.parse_args()
  return args


def run_psds_eval(dtc_threshold, gtc_threshold, alpha_ct, alpha_st):
  psds_eval = PSDSEval(ground_truth=groundtruth, metadata=metadata, dtc_threshold=dtc_threshold, gtc_threshold=gtc_threshold)
  psds_eval.clear_all_operating_points()
  predictions_per_threshold = glob.glob(os.path.join(args.predictions, 'threshold_*'))
  for i, predictions in enumerate(predictions_per_threshold):
    print(f"Adding Operating Point {i + 1:02d}/50", end="\r")
    threshold = float(predictions.split('_')[-1])
    det = pd.read_csv(os.path.join(predictions, 'outputs.tsv'), sep='\t')
    info = {"name": f"Op {i + 1:02d}", "threshold": threshold}
    psds_eval.add_operating_point(det, info=info)
  # compute the PSDS of the system represented by its operating points
  psds = psds_eval.psds(alpha_ct=alpha_ct, alpha_st=alpha_st, max_efpr=100)
  # plot the PSD-ROC and corresponding PSD-Score
  plot_psd_roc(psds)
  # compute intersection-based macro F-score
  det = pd.read_csv(os.path.join(args.predictions, 'threshold_0.51', 'outputs.tsv'), sep='\t')
  macro_f, class_f = psds_eval.compute_macro_f_score(det)
  print(f"macro F-score: {macro_f * 100:.2f}")
  for clsname, f in class_f.items():
    print(f"  {clsname}: {f * 100:.2f}")


if __name__ == '__main__':
  warnings.simplefilter('ignore')
  args = parse_args()
  groundtruth = pd.read_csv(args.groundtruth, sep='\t')
  metadata = pd.read_csv(args.metadata, sep='\t')
  # Evaluate PSDS for Scenario 1
  run_psds_eval(dtc_threshold=0.7, gtc_threshold=0.7, alpha_ct=0.0, alpha_st=1.0)
  # Evaluate PSDS for Scenario 2
  run_psds_eval(dtc_threshold=0.1, gtc_threshold=0.1, alpha_ct=0.5, alpha_st=1.0)