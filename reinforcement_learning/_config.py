hidden_size = 500
save_every = 1000
learning_rate = 0.0001
BATCH_SIZE = 256
GAMMA = 0.999
MAX_LENGTH = 30
max_turns_per_episode = 10
state_length = 4
retrain_discriminator_every = 1000

print_every = 25

teacher_force_ratio = 0.1

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = 30  # Maximum sentence length to consider