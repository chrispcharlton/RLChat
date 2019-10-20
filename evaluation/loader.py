from typing import Tuple, Text

from _requirements import *
from _config import *
from seq2seq.vocab import Voc

# from seq2seq.models import EncoderRNN, LuongAttnDecoderRNN
from pt_examples.functions import *


def load(load_path: Text) -> Tuple[object, object, object, object]:

    checkpoint = torch.load(load_path, map_location=device)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    voc = Voc(checkpoint['voc_dict']['name'])
    voc.__dict__ = checkpoint['voc_dict']

    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(embedding_sd)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    searcher = GreedySearchDecoder(encoder, decoder)

    return encoder, decoder, searcher, voc
