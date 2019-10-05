from _requirements import *
from seq2seq.vocab import MAX_LENGTH, SOS_token

class RLGreedySearchDecoder(nn.Module):

    def __init__(self, encoder, decoder, voc):
        super(RLGreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.voc = voc

    def forward(self, state, greedy=True, max_length=MAX_LENGTH):
        # Forward input through encoder model
        input_length = torch.LongTensor([len(s) for s in state])
        batch_size = state.size(0)
        encoder_outputs, encoder_hidden = self.encoder(state, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, batch_size, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            if greedy:
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            else:
                decoder_input = torch.randint(decoder_output.size(dim=1), (1, 1))
                decoder_scores = decoder_output.select(1, decoder_input)

            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens.view(batch_size, -1), all_scores.view(batch_size, -1)

    def response(self, state):
        tokens, scores = self(state)
        decoded_words = [self.voc.index2word[token.item()] for token in tokens[0]]
        return " ".join([x for x in decoded_words if not (x == 'EOS' or x == 'PAD')]), tokens
