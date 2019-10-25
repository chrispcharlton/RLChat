import os

# allow set up own rel save paths
BASE_DIR = os.path.dirname(__file__)

SAVE_PATH = 'data/save/cb_model/Alexa/2-2_500/'
SAVE_PATH_RL = 'data/save/rl_model/DQNseq2seq/'
SAVE_PATH_ADEM = 'data/save/adem_model/'
SAVE_PATH_DISCRIMINATOR = 'data/save/Adversarial_Discriminator/'
SAVE_PATH_SEQ2SEQ = 'data/save/cb_model/Alexa/2-2_500/'
RESULTS = 'data/results/'

dirs = [
    SAVE_PATH,
    SAVE_PATH_RL,
    SAVE_PATH_ADEM,
    SAVE_PATH_DISCRIMINATOR,
    SAVE_PATH_SEQ2SEQ,
    RESULTS,
]

for rel_path in dirs:
    abs_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(abs_path):
        os.makedirs(abs_path)

