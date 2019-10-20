import csv

from typing import List, Dict, Text

from constants import *
from evaluation.inference import get_response
from evaluation.loader import load

SAMPLE_SIZE = 25
DATA_SET = ''
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'results.csv')
SAVE_DIR = os.path.join(BASE_DIR, 'data/save')


def get_sample() -> List[str]:
    """ Returns a random sample of utterances from data """
    # temp placeholder list
    utterances = [
        'Hi how are you?',
        'Nice to see you',
        'Shocking weather today',
        'Here is another utterance',
        'Just another movie!',
        'Kids nowdays',
        'You',
    ]
    return utterances


def load_all() -> Dict[int, Dict]:
    """ 
    Loads and returns all of:
    - 1. seq2seq
    - 2. seq2seq with rl-train ADEM
    - 3. seq2seq with rl-train discrim
    - 4. seq2seq with rl-train both
    """
    all_models = {
        1: {
            'rel_path': 'cb_model/Alexa/2-2_500/4000_checkpoint.tar',
            'model': None,
        },
        # 2: {
        #     'rel_path': '',
        #     'model': None,
        # },
        # 3: {
        #     'rel_path': '',
        #     'model': None,
        # },
        4: {
            'rel_path': 'rl_model/DQNseq2seq/100_checkpoint.tar',
            'model': None,
        }
    }
    for i in range(1, 5):
        if i == 1:  # might need to load other models differently
            all_models[i]['model'] = load(os.path.join(SAVE_DIR, all_models[i]['rel_path']))
    return all_models


def inference_all(utterance: Text, models_set: Dict[int, Dict]) -> List[str]:
    """ """
    row = [utterance]
    for int_key, value in models_set.items():
        _, _, searcher, voc = value['model']
        result = get_response(searcher, voc, utterance)
        row.append(result)
        return row


def write_results(results_list: List[List]) -> None:
    """

    :type results_list: List
    """
    with open(RESULTS_FILE, mode='w') as f:
        res_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in results_list:
            res_writer.writerow(row)


if __name__ == "__main__":
    print('Obtaining sample of utterances...')
    samples = get_sample()
    print('Loading models...')
    models = load_all()  # returns a dict of key=int and val=dict
    results = []
    print('Evaluating...')
    for sample in samples:
        sample_result = inference_all(sample, models_set=models)
        results.append(sample_result)
    write_results(results)
    print(f'Complete. Results saved in {RESULTS_FILE}.')


