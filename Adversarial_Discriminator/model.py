from _requirements import *
from torch.autograd import Variable
from seq2seq import load_latest_state_dict
from seq2seq import Voc
from _config import *
from constants import *

def loadAdversarial_Discriminator(hidden_size=hidden_size, output_size=2, n_layers=1, dropout=0, path=SAVE_PATH_DISCRIMINATOR):
    state_dict = load_latest_state_dict(path)
    voc = Voc('placeholder_name')
    voc.__dict__ = state_dict['voc_dict']

    print('Building Adversarial_Discriminator model ...')
    embedding = nn.Embedding(voc.num_words, hidden_size)
    embedding.load_state_dict(state_dict['embedding'])
    embedding.to(device)
    model = Adversarial_Discriminator(hidden_size, output_size, embedding, n_layers, dropout)
    model.load_state_dict(state_dict['model'])
    return model

class Adversarial_Discriminator(nn.Module):
    def __init__(self, hidden_size, output_size, embedding, n_layers=1, dropout=0):
        super(Adversarial_Discriminator, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, state, hidden=None):
        # Convert word indexes to  embeddings
        # input_lengths = torch.LongTensor([len(s) for s in state], device=device)
        input_lengths = torch.tensor([len(s) for s in state], device=device, dtype=torch.long)
        embedded = self.embedding(state.t())

        batch_size = state.size(0)
        hidden = self._init_hidden(batch_size) if hidden is None else hidden

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

    def predict(self, state, hidden=None):
        '''
        Gets model output (tensors) and converts to numeric rating
        :param input_seq:
        :param input_lengths:
        :param hidden:
        :return: number 0 - 1 based on how human
        '''
        pred = self(state, hidden)
        return pred.data.max(1, keepdim=True)[1]

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*(1+int(self.gru.bidirectional)), batch_size, self.hidden_size, device=device)
        return Variable(hidden)

