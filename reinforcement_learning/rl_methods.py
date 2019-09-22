from _requirements import *
from seq2seq.vocab import MAX_LENGTH, SOS_token
from collections import namedtuple


def chat(searcher, env):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            reward, next_state, done = env.step(input_sentence)
            action = searcher.select_action(next_state)
            reward, next_state, done = env.step(action)
            print('Bot:', action)

        except KeyError:
            print("Error: Encountered unknown word.")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


BATCH_SIZE = 64
GAMMA = 0.999

class RLGreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, voc):
        super(RLGreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.voc = voc

    def forward(self, state, max_length=MAX_LENGTH):
        # Forward input through encoder model
        input_seq, input_length = state
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

    def select_action(self, state, max_length=MAX_LENGTH):
        """
        selects an action given state
        :param state:
        :param max_length:
        :return:
        """
        # global steps_done
        # sample = random.random()
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #     math.exp(-1. * steps_done / EPS_DECAY)
        # steps_done += 1
        # if sample > eps_threshold:
        #     with torch.no_grad():
        #         # t.max(1) will return largest column value of each row.
        #         # second column on max result is index of where max element was
        #         # found, so we pick action with the larger expected reward.
        #         return policy_net(state).max(1)[1].view(1, 1)
        # else:
        #     return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        ## TODO: add e-greedy?
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            tokens, scores = self(state, max_length)
        decoded_words = [self.voc.index2word[token.item()] for token in tokens]
        return " ".join([x for x in decoded_words if not (x == 'EOS' or x == 'PAD')])


def optimize_model(searcher, memory, en_optimizer, de_optimizer):
    if len(memory) < BATCH_SIZE:
        return
    else:
        print("Optimising...")

        transitions = memory.sample(BATCH_SIZE)

        est = []
        actual = []

        for n in transitions:
        # Compute Q(s_t) - Q_value of the starting state for transition n
        # searcher outputs tensor of value for each word in the action sequence
        # max value is expected reward (maybe switch to average value?)
            est.append(searcher(n.state)[1].max())

            # if s_t is terminal then true reward is the reward given by environment
            # if not then sum actual reward with Q value of expected future reward
            if n.done:
                q = n.reward
            else:
                q_next_state = searcher(n.state)[1].max()
                q = (q_next_state * GAMMA) + n.reward

            actual.append(q)

        # Compute Huber loss
        est = torch.stack(est)
        actual = torch.stack(actual)
        loss = F.smooth_l1_loss(est, actual)
        loss.backward()
        print ("loss =", loss)

        # Optimize the model
        en_optimizer.zero_grad()
        de_optimizer.zero_grad()
        ## Not sure below is necessary
        # for param in searcher.encoder.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # for param in searcher.decoder.parameters():
        #     param.grad.data.clamp_(-1, 1)
        en_optimizer.step()
        de_optimizer.step()

        return loss


# def optimize_model(searcher, memory, en_optimizer, de_optimizer):
#     if len(memory) < BATCH_SIZE:
#         return
#     transitions = memory.sample(BATCH_SIZE)
#     # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#     # detailed explanation). This converts batch-array of Transitions
#     # to Transition of batch-arrays.
#     batch = Transition(*zip(*transitions))
#
#     # Compute a mask of non-final states and concatenate the batch elements
#     # (a final state would've been the one after which simulation ended)
#
#     #below line dtype changed to bool form uint8 - deprecated error message
#     non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                           batch.next_state)), device=device, dtype=torch.bool)
#
#     #changed cat to stack in below
#     non_final_next_states = torch.cat([s[0] for s in batch.next_state
#                                                 if s is not None])
#
#     state_batch1 = torch.stack([s[0] for s in batch.state]) #changed cat to stack
#     action_batch = torch.cat(batch.action)
#     reward_batch = torch.cat(batch.reward)
#     # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#     # columns of actions taken. These are the actions which would've been taken
#     # for each batch state according to policy_net
#
#     state_action_values = searcher((state_batch1, state_batch2))
#
#     # Compute V(s_{t+1}) for all next states.
#     # Expected values of actions for non_final_next_states are computed based
#     # on the "older" target_net; selecting their best reward with max(1)[0].
#     # This is merged based on the mask, such that we'll have either the expected
#     # state value or 0 in case the state was final.
#     next_state_values = torch.zeros(BATCH_SIZE, device=device)
#     next_state_values[non_final_mask] = searcher(non_final_next_states).max(1)[0].detach()
#     # Compute the expected Q values
#
#     expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#     # Compute Huber loss
#     loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
#
#     # Optimize the model
#     en_optimizer.zero_grad()
#     de_optimizer.zero_grad()
#     loss.backward()
#     for param in searcher.encoder.parameters():
#         param.grad.data.clamp_(-1, 1)
#     for param in searcher.decoder.parameters():
#         param.grad.data.clamp_(-1, 1)
#     en_optimizer.step()
#     de_optimizer.zero_grad()
#
