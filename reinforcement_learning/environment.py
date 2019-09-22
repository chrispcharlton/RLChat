from rl_methods import *
from seq2seq import indexesFromSentence


class Env(object):
    def __init__(self, voc, state_length=1):
        self.voc = voc
        self.state_length = state_length
        self.reset()

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
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        return input_batch, lengths

    def reset(self):
        self._state = [" ".join(['hello'])] * self.state_length

    def step(self, action):
        reward = self.calculate_reward(action)
        state = [action] + self._state[:-1]
        state = self.state2tensors(state)
        done = self.is_done(state)
        return reward, state, done

    def calculate_reward(self, action):
        return round(0.1 * len(action.replace('.', '').split()), 2)

    def is_done(self, state):
        return random.random() > 0.7
