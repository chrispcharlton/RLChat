from Adversarial_Discriminator import loadAdversarial_Discriminator
from _requirements import *
from ADEM import loadADEM
from seq2seq import indexesFromSentence
from reinforcement_learning._config import MAX_LENGTH, max_turns_per_episode, state_length

from data.amazon.dataset import standardise_sentence, AlexaDataset



def chat(policy, env):
    env.reset()
    env._state = []

    while 1:
        try:
            input_sentence = input('> ')
            if input_sentence == 'q':
                break
            input_sentence = env.sentence2tensor(standardise_sentence(input_sentence))
            env.update_state(input_sentence)
            response, tensor = policy.response(env.state)
            env.update_state(tensor)
            print('Bot:', response)
        except KeyError:
            print("Error: Encountered unknown word.")

def pad_with_zeroes(seq):
    state_tensor = torch.zeros((1, MAX_LENGTH), device=device).long()
    # state_tensor[1, :len(seq)] = torch.LongTensor(seq)
    state_tensor[1, :len(seq)] = torch.tensor(seq, device=device, dtype=torch.long)
    return state_tensor

class Env(object):
    def __init__(self, voc, dataset, state_length=state_length):
        print('Initialising Environment...')
        self.voc = voc
        self.state_length = state_length
        self.dataset = dataset
        self.adem = loadADEM()
        self.AD = loadAdversarial_Discriminator()
        self.n_turns = 1
        self.user_sim_model = None
        self.reset()

    @property
    def state(self):
        return torch.cat(self._state, 1)

    def state_of_len(self, n):
        n = len(self._state) if n > len(self._state) else n
        state_seq = torch.cat(self._state[-n:], 1)
        return state_seq

    def update_state(self, tensor):
        if len(self._state) >= self.state_length:
            self._state.pop(0)
        self._state.append(tensor)

    def reset(self, input_sentence=None):
        input_sentence = self.dataset.random_opening_line() if input_sentence is None else input_sentence
        self._state = [self.sentence2tensor(input_sentence)]
        self.n_turns = 1

    def sentence2tensor(self, sentence):
        ### Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(self.voc, sentence)]
        seq = torch.tensor(indexes_batch, device=device, dtype=torch.long)
        # Use appropriate device
        seq = seq.to(device)
        return seq

    def user_sim(self, state):
        if self.user_sim_model:
            # take [0] as user_sim_model returns (action, probability)
            return self.user_sim_model(state)[0]
        else:
            return self.sentence2tensor(" ".join(['hello']))

    def step(self, action, teacher_response=None):
        self.n_turns += 2
        self.update_state(action)
        reward = self.calculate_reward(self.state_of_len(2)) # if teacher_response is None else float(1)
        done = self.is_done()
        if not done:
            next_utterance = self.user_sim(self.state) if teacher_response is None else self.sentence2tensor(teacher_response)
            self.update_state(next_utterance)
            next_state = self.state
        else:
            next_state = None
        return reward, next_state, done

    def calculate_reward(self, next_state):
        # TODO: reward should probably be a vector of whole sentence, with reward for each token
        # return 0.5 * (float(self.adem.predict(next_state).item() / 4) + float(self.AD(next_state))) # Both
        # return float(self.AD(next_state)) # Discriminator only
        return (float(self.adem.predict(next_state).item() / 4)) # ADEM only

    def is_done(self):
        return (len(set(self._state)) != len(self._state)) or (self.n_turns >= max_turns_per_episode)
