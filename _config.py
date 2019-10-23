from _requirements import device

attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 256 if device == 'cuda' else 64
learning_rate = 0.0001
decoder_learning_ratio = 5.0

save_every = 500

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = 30  # Maximum sentence length to consider