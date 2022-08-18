# Options for audio extraction
sample_rate = 16000
loudness_normalize = True

# Type of input features, one of ['mel', 'stft', 'gammatone']
features = 'mel'

# Options for spectrogram extraction
n_fft = 2048
frame_step_ms = 24
top_db = None

# Options for mel spectrogram (or gammatonegram) extraction
num_mel_bins = 128
fmin_hz = 0
fmax_hz = None

# Model options
batch_size = 32
learning_rate = 5e-4
state_size = 256
variational_encoder = True
num_latents = state_size
synthetic_loss_weight = 1.0
kld_weight = 1e-4

# Training options
test_size = 0.05  # If no test data are specified, withhold this % of training data as test data

# Classification options
threshold = 0.5
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
