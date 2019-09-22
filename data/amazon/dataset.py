from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
import random
import json
import os

Pair = namedtuple('Pair', ('utterance', 'response', 'rating', 'conversation_id'))
numeric_ratings = {'Poor':1, 'Not Good':2, 'Passable':3, 'Good':4, 'Excellent':5}


def load_alexa_pairs(fname='train.json', dir='./data/amazon'):
    pairs = []
    data = json.loads(open(os.path.join(dir, fname), 'r').read())
    for c_id, conversation in data.items():
        for utterance, response in zip(conversation['content'][:-1],conversation['content'][1:]):
            if response['turn_rating'] != '':
                pairs.append(Pair(utterance=utterance['message'], response=response['message'], rating=numeric_ratings[response['turn_rating']], conversation_id=c_id))
    return pairs

class AlexaDataset(Dataset):
    """Amazon Alexa Conversations dataset."""

    def __init__(self, json, dir='./data/amazon'):
        """
        Args:
            json (string): Name of json file to load
            dir (string): Directory where json is stored
        """
        self.data = load_alexa_pairs(json, dir)
        self.ids = list(set([p.conversation_id for p in self.data]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_conversation(self, id):
        return [p for p in self.data if p.conversation_id == id]

    def random_conversation(self):
        id = random.choice(self.ids)
        return [p for p in self.data if p.conversation_id == id]

if __name__ == '__main__':
    data = AlexaDataset('train.json')
    print(len(data))
    print(data[0])
    c = data.get_conversation('t_bde29ce2-4153-4056-9eb7-f4ad710505fe')
    s = data.random_conversation()