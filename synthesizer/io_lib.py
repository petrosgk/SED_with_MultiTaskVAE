import librosa
import soundfile
import synthesizer.hparams as opt


def load_audio_data(audio_file):
  # Load audio file
  audio_data, _ = librosa.core.load(path=audio_file, sr=opt.sample_rate)
  return audio_data


def write_audio_data(audio_data, filename):
  # Save audio file
  soundfile.write(filename, audio_data, samplerate=opt.sample_rate)