from _config import MAX_LENGTH
from torch.utils.data import Dataset
from collections import namedtuple, Counter
import random
import json
import os


Pair = namedtuple('Pair', ('utterance', 'response', 'rating', 'conversation_id', 'opening_line'))
numeric_ratings = {'Poor':0, 'Not Good':1, 'Passable':2, 'Good':3, 'Excellent':4}


def standardise_sentence(sentence):
    sentence = sentence.replace(',',' , ').replace('.',' . ').replace('?',' ? ').replace('!',' ! ').replace('(',' ( ').replace(')',' ) ').replace('%',' % ')
    while '  ' in sentence:
        sentence = sentence.replace('  ',' ')
    return sentence.lower()

def load_alexa_pairs(fname='train.json', dir='./data/amazon'):
    pairs = []
    data = json.loads(open(os.path.join(dir, fname), 'r').read())
    for c_id, conversation in data.items():
        first = True
        for utterance, response in zip(conversation['content'][:-1],conversation['content'][1:]):
            u_sentence, r_sentence = standardise_sentence(utterance['message']), standardise_sentence(response['message'])
            if response['turn_rating'] not in [''] and len(u_sentence.split(' ')) < MAX_LENGTH and len(r_sentence.split(' ')) < MAX_LENGTH:
                pairs.append(Pair(utterance=u_sentence,
                                  response=r_sentence,
                                  rating=numeric_ratings[response['turn_rating']],
                                  conversation_id=c_id,
                                  opening_line=first))
                first = False
    return pairs


class AlexaDataset(Dataset):
    """Amazon Alexa Conversations dataset."""

    def __init__(self, json=None, dir='./data/amazon', rare_word_threshold=0):
        """
        Args:
            json (string): Name of json file to load
            dir (string): Directory where json is stored
            rare_word_threshold: Remove pairs which contain words that appear less than or equal to threshold
        """
        self.data = []
        self.rare_word_threshold = rare_word_threshold
        if json is not None:
            self.add_pairs_from_json(json, dir)
        else:
            for f in [f for f in os.listdir(dir) if f.endswith('.json')]:
                print("Added {} to dataset".format(f))
                self.add_pairs_from_json(f, dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @property
    def conversation_ids(self):
        return set([p.conversation_id for p in self.data])

    @property
    def opening_lines(self):
        return [self.get_conversation(c)[0].utterance for c in self.conversation_ids]

    def balance_data(self):
        min_obs = 10000000
        for r in numeric_ratings.values():
            obs = len([p for p in self.data if p.rating==r])
            min_obs = obs if obs < min_obs else min_obs
        new_data = []
        for r in numeric_ratings.values():
            r_data = [p for p in self.data if p.rating==r]
            if len(r_data) >= min_obs:
                r_data = r_data[:min_obs]
                new_data += r_data
        self.data = new_data
        self.ids = list(set([p.conversation_id for p in self.data]))

    def add_pairs_from_json(self, json, dir='./data/amazon'):
        self.data += load_alexa_pairs(json, dir)
        self._trim_rare_words(self.rare_word_threshold)
        self.ids = list(set([p.conversation_id for p in self.data]))

    def random_opening_line(self):
        return random.choice([p.utterance for p in self.data if p.opening_line])

    def get_conversation(self, id):
        return [p for p in self.data if p.conversation_id == id]

    def random_conversation(self):
        id = 0
        while id == 0:
            id = random.choice(self.ids)
        return [p for p in self.data if p.conversation_id == id]

    def _rare_words(self, threshold=3):
        count = Counter()
        for pair in self.data:
            for word in set(pair.utterance.split(' ')):
                count[word] += 1
        rare_words = set([word for word, cnt in count.items() if cnt <= threshold])
        return rare_words

    def _trim_rare_words(self, threshold=3):
        rare_words = self._rare_words(threshold)
        new_data = []
        for pair in self.data:
            words = pair.utterance.split(' ') + pair.response.split(' ')
            if set(words).isdisjoint(rare_words):
                new_data.append(pair)
        print('{} pairs trimmed, {} remain'.format(len(self.data) - len(new_data), len(new_data)))
        self.data = new_data

    def trimPairsToVocab(self, voc):
        keepPairs = []
        for pair in self.data:
            keep = True
            for word in pair.utterance.split(' ') + pair.response.split(' '):
                if word not in voc.word2index:
                    keep = False
            if keep:
                keepPairs.append(pair)
        print('=============={}=============='.format(len(self.data)-len(keepPairs)))
        self.data = keepPairs

    def add_scrambled_training_data(self, addition_rate=0.2, rating='Poor'):
        pairs = []
        for i in range(int(len(self.data) * addition_rate)):
            utterance = random.choice(self.data).utterance
            response = random.choice(self.data).response
            pairs.append(Pair(utterance=utterance,
                              response=response,
                              rating=numeric_ratings[rating],
                              conversation_id='0',
                              opening_line=False))
        self.data += pairs

if __name__ == '__main__':
    data = AlexaDataset(rare_word_threshold=2)
    print(len(data))
    print(data[0])
    c = data.get_conversation('t_bde29ce2-4153-4056-9eb7-f4ad710505fe')
    s = data.random_conversation()
    data.add_scrambled_training_data()
