from reinforcement_learning.rl_methods import *
from ADEM import loadADEM
from seq2seq import indexesFromSentence

max_turns_per_episode = 10

class Env(object):
    def __init__(self, voc, state_length=4):
        print('Initialising Environment...')
        self.voc = voc
        self.state_length = state_length
        self.reset()
        self.adem = loadADEM()
        self.n_turns = 1

    @property
    def state(self):
        return torch.cat(self._state, 1)

    def update_state(self, tensor):
        if len(self._state) >= self.state_length:
            self._state.pop(0)
        self._state.append(tensor)

    def reset(self):
        self._state = [self.state2tensors([" ".join(['hello'])])] * self.state_length
        self.n_turns = 1

    def state2tensors(self, state):
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(self.voc, s) for s in state]
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch) #.transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        return input_batch

    def user_sim(self, state):
        return self.state2tensors([" ".join(['hello'])])

    def step(self, action):
        self.n_turns += 2
        self.update_state(action)
        reward = self.calculate_reward(self.state)
        self.update_state(self.user_sim(self.state))
        done = self.is_done(self.state)
        next_state = self.state
        if done:
            next_state = None
        return reward, next_state, done

    def calculate_reward(self, next_state):
        # TODO: reward should probably be a vector of whole sentence, with reward for each token
        return float(self.adem.predict(next_state))

    def is_done(self, state):
        return self.n_turns >= max_turns_per_episode
