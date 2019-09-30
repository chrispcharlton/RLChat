from reinforcement_learning.rl_methods import *
from ADEM import loadADEM
from seq2seq import indexesFromSentence
from ADEM import model

max_turns_per_episode = 10

class Env(object):
    def __init__(self, voc, state_length=1):
        print('Initialising Environment...')
        self.voc = voc
        self.state_length = state_length
        self.reset()
        self.adem = loadADEM()
        self.n_turns = 1

    @property
    def state(self):
        return self.state2tensors(self._state)

    def state2tensors(self, state):
        ### Format input sentence as a batch
        # words -> indexes
        # disgusting padding
        indexes_batch = [indexesFromSentence(self.voc, s) for s in state]
        # flat_batch = [[i for sub in indexes_batch for i in sub]]
        # Create lengths tensor
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch) #.transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        return input_batch

    def reset(self):
        self._state = [" ".join(['hello'])] * self.state_length
        self.n_turns = 1

    def step(self, action):
        self.n_turns += 2
        next_state = [action] + self._state[:-1]
        next_state = self.state2tensors(next_state)
        reward = self.calculate_reward(next_state)
        done = self.is_done(next_state)
        if done:
            next_state = None
        return reward, next_state, done

    def calculate_reward(self, next_state):
        # TODO: reward should probably be a vector of whole sentence, with reward for each token
        return float(self.adem.predict(next_state))

    def is_done(self, state):
        return self.n_turns >= max_turns_per_episode
