from _requirements import *
from torch.autograd import Variable

class DQN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(DQN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, state, hidden=None):
        # Convert word indexes to  embeddings
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

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers*(1+int(self.gru.bidirectional)), batch_size, self.hidden_size, device=device)
        return Variable(hidden)

