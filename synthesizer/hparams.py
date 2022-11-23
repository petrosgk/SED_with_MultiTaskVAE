# Options for audio extraction
sample_rate = 16000
loudness_normalize = True

# Type of input features, one of ['mel', 'stft', 'gammatone']
features = 'mel'

# Options for spectrogram extraction
n_fft = 2048
frame_step_ms = 12
top_db = None

# Options for mel spectrogram (or gammatonegram) extraction
num_mel_bins = 128
fmin_hz = 0
fmax_hz = None

# Model options
learning_rate = 1e-4
state_size = 256
num_latents = 8
kld_weight = 1e-4

# Inference options
infer_freq_epochs = 10
num_iterations = 10
num_perturbations = 5

# Training options
test_size = 0.05  # If no test data are specified, withhold this % of training data as test data

# Class labels
labels = ['Speech',
          'Dog',
          'Cat']