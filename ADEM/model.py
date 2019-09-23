from _requirements import *
from torch.autograd import Variable
import numpy as np
from seq2seq import loadModel
from seq2seq import loadAlexaData, batch2TrainData
from _config import *

class ADEM(nn.Module):
    def __init__(self, hidden_size, output_size, embedding, n_layers=1, dropout=0):
        super(ADEM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, input_lengths, hidden=None):
        # batch_size = input_seq.size(0)
        # input_seq = input_seq.t()
        # # Convert word indexes to embeddings
        # embedded = self.embedding(input_seq)
        # hidden = self._init_hidden(batch_size)
        # output, hidden = self.gru(embedded, hidden)
        # fc_output = self.fc(hidden)
        # return fc_output

        # Convert word indexes to  embeddings
        embedded = self.embedding(input_seq)

        batch_size = input_seq.size(0)
        # hidden = self._init_hidden(batch_size) if hidden is None else hidden

        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        self.gru.flatten_parameters()
        output, hidden = self.gru(packed, hidden)

        output = self.fc(hidden[-1])
        # Unpack padding
        # outputs, _ = nn.utils.rnn.pad_packed_sequence(fc_output)
        # Sum bidirectional GRU outputs
        # outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return output

    def predict(self, input_seq, input_lengths, hidden=None):
        '''
        Gets model output (tensors) and converts to numeric rating
        :param input_seq:
        :param input_lengths:
        :param hidden:
        :return: number 0 - 4 corresponding to rating from Alexa dataset
        '''
        pred = self(input_seq, input_lengths, hidden)
        return pred.data.max(1, keepdim=True)[1]

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return Variable(hidden)
