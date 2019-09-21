from requirements import *
from seq2seq.models import EncoderRNN, LuongAttnDecoderRNN
from seq2seq.vocab import Voc

from seq2seq.rl_methods import *
from seq2seq.prepareTrainData import EOS_token


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

def load_latest_state_dict():

    savepath = 'data\\save\\cb_model\\cornell movie-dialogs corpus\\2-2_500'
    try:
        saves = os.listdir(savepath)
    except FileNotFoundError:
        savepath = os.path.join("C:\\Users\\Christopher\\PycharmProjects\\RLChat", savepath)
        saves = os.listdir(savepath)
    max_save = saves[0]
    for save in saves:
         if int(save.split('_')[0]) > int(max_save.split('_')[0]):
             max_save = save
    return torch.load(open(os.path.join(savepath, max_save), 'rb'))

model = load_latest_state_dict()

attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

learning_rate = 0.0001
decoder_learning_ratio = 5.0

voc = Voc(model['voc_dict']['name'])
voc.__dict__ = model['voc_dict']

embedding = nn.Embedding(voc.num_words, hidden_size)

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

encoder.load_state_dict(model['en'])
decoder.load_state_dict(model['de'])

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

searcher = RLGreedySearchDecoder(encoder, decoder, voc)


memory = ReplayMemory(1000)

# RL training loop
num_episodes = 50
env = Env(voc)
for i_episode in range(num_episodes):
    print("Episode", i_episode+1)
    env.reset()
    state = env.state
    done = False
    while not done:
        action = searcher.select_action(state)
        reward, next_state, done = env.step(action)
        reward = torch.tensor([reward], device=device)
        action = env.state2tensors([action])[0]
        memory.push(state, action, next_state, reward, done)
        state = next_state
    optimize_model(searcher, memory, encoder_optimizer, decoder_optimizer)
        # if done:
            # record episode duration?

    # TODO: implement target/policy net?
    # if i_episode % TARGET_UPDATE == 0:
    #     target_net.load_state_dict(policy_net.state_dict())

