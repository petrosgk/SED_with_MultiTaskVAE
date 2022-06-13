import argparse
import os
import glob


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--path_to_transcripts', required=True, type=str, help='Path to transcripts.')
  args = parser.parse_args()
  return args


def check_transcript_file(transcript_file):
  prev_offset = None
  prev_event_label = None
  with open(transcript_file, 'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
      onset, offset, event_label = line.rstrip('\n').split('\t')
      if idx > 0:  # Skip the 1st line
        if event_label == prev_event_label:
          if float(onset) < prev_offset:
            return True
      prev_offset = float(offset)
      prev_event_label = event_label
  return False


if __name__ == '__main__':
  args = parse_args()
  transcripts = glob.glob(os.path.join(args.path_to_transcripts, '*.txt'))
  num_problem_transcripts = 0
  for transcript in transcripts:
    if check_transcript_file(transcript):
      print(transcript)
      num_problem_transcripts += 1
  print('Found %d transcripts with overlapping event labels.' % num_problem_transcripts)
