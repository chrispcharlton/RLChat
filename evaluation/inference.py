import torch

from typing import Text

from _requirements import *

from seq2seq.vocab import normalizeString
from seq2seq.prepareTrainData import indexesFromSentence

MAX_LENGTH = 10

def evaluate(searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def get_response(searcher, voc, utterance) -> Text:
    """ Return response from given model searcher, vocab per utterance"""
    try:
        input_sentence = normalizeString(utterance)
        output_words = evaluate(searcher, voc, input_sentence)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        return ' '.join(output_words)
    except KeyError:
        return 'Error: Encountered unknown word.'

