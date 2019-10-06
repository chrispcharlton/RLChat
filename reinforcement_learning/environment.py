from _requirements import *
from ADEM import loadADEM
from seq2seq import indexesFromSentence
from reinforcement_learning._config import MAX_LENGTH, max_turns_per_episode, state_length


def chat(policy, env):
    input_sentence = ''
    env.reset()
    env._state = []

    while(1):
        try:
            input_sentence = input('> ')
            if input_sentence == 'q':
                break
            input_sentence = env.sentence2tensor(input_sentence)
            env.update_state(input_sentence)
            response, tensor = policy.response(env.state)
            env.update_state(tensor)
            print('Bot:', response)
        except KeyError:
            print("Error: Encountered unknown word.")


def pad_with_zeroes(seq):
    state_tensor = torch.zeros((1, MAX_LENGTH)).long()
    state_tensor[1, :len(seq)] = torch.LongTensor(seq)
    return state_tensor

class Env(object):
    def __init__(self, voc, state_length=state_length):
        print('Initialising Environment...')
        self.voc = voc
        self.state_length = state_length
        self.reset()
        self.adem = loadADEM()
        self.n_turns = 1
        self.user_sim_model = None

    @property
    def state(self):
        return torch.cat(self._state, 1)

    def update_state(self, tensor):
        if len(self._state) >= self.state_length:
            self._state.pop(0)
        self._state.append(tensor)

    def reset(self, input_sentence=None):
        input_sentence = " ".join(['hello']) if input_sentence is None else input_sentence
        self._state = [self.sentence2tensor(input_sentence)]
        self.n_turns = 1

    def sentence2tensor(self, sentence):
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(self.voc, sentence)]
        # Transpose dimensions of batch to match models' expectations
        seq = torch.LongTensor(indexes_batch) #.transpose(0, 1)
        # Use appropriate device
        seq = seq.to(device)
        return seq

    def user_sim(self, state):
        if self.user_sim_model:
            return self.user_sim_model(state)[0]
        else:
            return self.sentence2tensor(" ".join(['hello']))

    def step(self, action):
        self.n_turns += 2
        self.update_state(action)
        reward = self.calculate_reward(self.state)
        done = self.is_done()
        if not done:
            self.update_state(self.user_sim(self.state))
            next_state = self.state
        else:
            next_state = None
        return reward, next_state, done

    def calculate_reward(self, next_state):
        # TODO: reward should probably be a vector of whole sentence, with reward for each token
        return float(self.adem.predict(next_state) / 4)

    def is_done(self):
        return (len(set(self._state)) != len(self._state)) or (self.n_turns >= max_turns_per_episode)
