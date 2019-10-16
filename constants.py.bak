import os

# allow set up own rel save paths
BASE_DIR = os.path.dirname(__file__)

SAVE_PATH = 'data/save/cb_model/cornell movie-dialogs corpus/2-2_500/'
SAVE_PATH_RL = 'data/save/rl_model/DQNseq2seq/'
SAVE_PATH_ADEM = 'data/save/adem_model/'

dirs = [
    SAVE_PATH,
    SAVE_PATH_RL,
    SAVE_PATH_ADEM,
]

for rel_path in dirs:
    abs_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)

