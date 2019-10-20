from _config import MAX_LENGTH
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from seq2seq.vocab import *
from seq2seq.processText import *
import random
import json
import os

Pair = namedtuple('Pair', ('utterance', 'response', 'rating', 'conversation_id'))
numeric_ratings = {'Poor':0, 'Not Good':1, 'Passable':2, 'Good':3, 'Excellent':4}
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

class CornellDataset(Dataset):
    """Cornell Conversations dataset."""

    def __init__(self, filename, voc=None, isTrainSet=True, testSetFraction = 0.1):
        """
        Args:
            json (string): Name of the file to load
            dir (string): Directory where the file is stored
        """

        # self.data, self.voc = load_pairs(filename, dir_path)
        self.data, self.voc = load_cornell_pairs(filename, dir_path)

        if(not voc == None):
            self.data = trimPairsToVocab(self.data, voc)
            self.voc = voc
        if(isTrainSet):
            self.data = self.data[: len(self.data) - math.ceil(len(self.data)*testSetFraction)]
        else:
            self.data = self.data[len(self.data) - math.ceil(len(self.data)*testSetFraction) :]
        # self.ids = list(set([p.conversation_id for p in self.data]))  # temp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def get_conversation(self, id):
        return [p for p in self.data if p.conversation_id == id]

    def random_conversation(self):
        id = random.choice(self.ids)
        return [p for p in self.data if p.conversation_id == id]
    def get_voc(self):
        return self.voc



def trimPairsToVocab(pairs, voc):
    keepPairs = []

    for pair in pairs:
        keep = True
        for word in pair.utterance.split(' ')+pair.response.split(' '):
            if word not in voc.word2index:
                keep = False
                break
        if(keep):
            keepPairs.append(pair)
    return keepPairs

def load_pairs(fname='formatted_movie_lines.txt', dir='./data/movie_dialogs'):
    # Load/Assemble voc and pairs
    save_dir = os.path.join("data", "save")

    corpus_name = "movie_dialogs"
    corpus = os.path.join("..\\data", corpus_name)
    datafile = os.path.join(dir_path, "formatted_movie_lines.txt")
    voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    MIN_COUNT = 3  # Minimum word count threshold for trimming

    # Trim voc and pairs
    voc, pairs = trimRareWords(voc, pairs, MIN_COUNT)
    return pairs, voc


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return voc, keep_pairs

if __name__ == '__main__':
    data = CornellDataset()
    print(len(data))
    print(data[0])
    s = data.random_conversation()