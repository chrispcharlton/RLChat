from _requirements import *
from _config import *
from seq2seq.loader import loadModel, saveStateDict
from seq2seq.vocab import Voc
from reinforcement_learning.qnet import DQN
from reinforcement_learning._config import save_every, hidden_size, learning_rate, BATCH_SIZE, GAMMA, retrain_discriminator_every, print_every, teacher_force_ratio
from reinforcement_learning.environment import Env, chat
from reinforcement_learning.model import RLGreedySearchDecoder
from Adversarial_Discriminator.train import trainAdversarialDiscriminatorOnLatestSeq2Seq
from collections import namedtuple
from numpy import mean
from constants import *
from data.amazon.dataset import AlexaDataset
from torch.utils.data import DataLoader

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done', 'prob'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def seqs_to_padded_tensors(seqs, max_length=None):
    ## TODO: does this need to pad with spaces (token) insteade of 0s?
    # lengths = torch.LongTensor([len(s) if s is not None else 0 for s in seqs], device=device)
    lengths = torch.tensor([len(s) if s is not None else 0 for s in seqs], device=device, dtype=torch.long)
    max_length = max_length if max_length is not None else lengths.max()
    state_tensor = torch.zeros((len(seqs), max_length), device=device).long()
    for idx, (seq, seq_len) in enumerate(zip(seqs, lengths)):
        if seq is not None:
            # state_tensor[idx, :seq_len] = torch.LongTensor(seq)
            state_tensor[idx, :seq_len] = torch.tensor(seq, device=device, dtype=torch.long)
    return state_tensor, lengths


def optimise_qnet(state_action_values, expected_state_action_values, qnet, qnet_optimizer, retain_graph=True):
    # Compute MSE loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)
    qnet.zero_grad()
    loss.backward(retain_graph=retain_graph)
    qnet_optimizer.step()
    return loss


def qloss(probs, q_values):
    return torch.mean(torch.mul(torch.log(probs), q_values).mul(-1), -1).mean()


def optimize_batch_q(policy, qnet, qnet_optimizer, memory, en_optimizer, de_optimizer):

    ## TODO: clean up this function

    if len(memory) < BATCH_SIZE:
        return None, None

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Convert batch to stacked tensors to input into model and sort by state length
    sa_pairs = [torch.cat((s[0], a[0])) for s, a in zip(batch.state, batch.action)]
    states, state_lengths = seqs_to_padded_tensors([s for s in sa_pairs])
    next_states, next_state_lengths = seqs_to_padded_tensors([s[0] if s is not None else s for s in batch.next_state])

    state_lengths, perm_idx = state_lengths.sort(0, descending=True)
    states = states[perm_idx]
    next_states = next_states[perm_idx]

    reward_batch = torch.stack(batch.reward)
    reward_batch = reward_batch[perm_idx]

    prob_batch = torch.stack(batch.prob)
    prob_batch = prob_batch[perm_idx]

    # Compute a mask of non-final states. Mask is used to allocate 0 future reward to terminal states.
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple([True if s.sum() != 0 else False for s in next_states]), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in next_states if s.sum() != 0])

    # Compute Q(s_t, a) - the model computes Q(s_t).
    state_action_values = qnet(states)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based on the "older" target_net
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, 1, device=device)
    next_state_values[non_final_mask] = qnet(non_final_next_states)

    # Compute the expected Q values for next states
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    dqn_loss = optimise_qnet(state_action_values, expected_state_action_values, qnet, qnet_optimizer)

    policy_loss = qloss(prob_batch, expected_state_action_values)
    en_optimizer.zero_grad()
    de_optimizer.zero_grad()
    policy_loss.backward()
    en_optimizer.step()
    de_optimizer.step()

    return dqn_loss, policy_loss

def teacher_force_ep(env, memory, policy, qnet, qnet_optimizer, encoder_optimizer, decoder_optimizer):
    conv = []
    while len(conv) < 2:
        conv = env.dataset.random_conversation()
    pair = conv.pop(0)
    env.reset(pair.utterance)
    state = env.state
    done = False
    length = 0
    ep_reward = 0
    ep_q_loss = []
    while not done:
        pair = conv.pop(0)
        length += 1
        action, prob = env.sentence2tensor(pair.utterance), 1.00
        prob = torch.tensor([prob], device=device)
        reward, next_state, done = env.step(action, teacher_response=pair.response)
        ep_reward += reward
        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward, done, prob)
        state = next_state
        dqn_loss, policy_loss = optimize_batch_q(policy, qnet, qnet_optimizer, memory, encoder_optimizer,
                                                 decoder_optimizer)
        dqn_loss = dqn_loss.item() if dqn_loss is not None else 0
        ep_q_loss.append(dqn_loss)
        if len(conv) > 1:
            conv.pop(0)
        else:
            done = True
    ep_q_loss = mean(ep_q_loss)
    return ep_q_loss, ep_reward, policy_loss

def model_ep(env, memory, policy, qnet, qnet_optimizer, encoder_optimizer, decoder_optimizer):
    env.reset()
    state = env.state
    done = False
    length = 0
    ep_reward = 0
    ep_q_loss = []
    while not done:
        length += 1
        action, prob = policy(state)
        prob = torch.tensor([torch.mean(prob)], device=device)
        reward, next_state, done = env.step(action)
        ep_reward += reward
        reward = torch.tensor([reward], device=device)
        memory.push(state, action, next_state, reward, done, prob)
        state = next_state
        dqn_loss, policy_loss = optimize_batch_q(policy, qnet, qnet_optimizer, memory, encoder_optimizer,
                                                 decoder_optimizer)
        dqn_loss = dqn_loss.item() if dqn_loss is not None else 0
        ep_q_loss.append(dqn_loss)
    ep_q_loss = mean(ep_q_loss)
    return ep_q_loss, ep_reward, policy_loss

def train(load_dir=SAVE_PATH, save_dir=SAVE_PATH_RL, num_episodes=10000, env=None, teacher_force_ratio=teacher_force_ratio):
    episode, encoder, decoder, encoder_optimizer, decoder_optimizer, voc = loadModel(directory=load_dir)
    voc = Voc.from_dataset(AlexaDataset())
    policy = RLGreedySearchDecoder(encoder, decoder, voc)
    embedding = nn.Embedding(voc.num_words, hidden_size)
    qnet = DQN(hidden_size, embedding).to(device)
    qnet_optimizer = torch.optim.Adam(qnet.parameters(), lr=learning_rate)
    memory = ReplayMemory(1000)
    env = env if env else Env(voc, AlexaDataset())

    # set episode number to 0 if starting from warm-started model. If loading rl-trained model continue from current number of eps
    if "/rl_models/" not in load_dir:
        episode = 0

    total_rewards = []
    dqn_losses = []

    ad_data = DataLoader(env.dataset, batch_size=BATCH_SIZE, shuffle=True)

    # RL training loop
    print("Training for {} episodes...".format(num_episodes))

    for i_episode in range(1, num_episodes+1):
        if i_episode % 10 == 0:
            env.user_sim_model = policy
        if random.random() < teacher_force_ratio:
            ep_q_loss, ep_reward, policy_loss = teacher_force_ep(env, memory, policy, qnet, qnet_optimizer, encoder_optimizer, decoder_optimizer)
        else:
            ep_q_loss, ep_reward, policy_loss = model_ep(env, memory, policy, qnet, qnet_optimizer, encoder_optimizer, decoder_optimizer)
        total_rewards.append(ep_reward)
        dqn_losses.append(ep_q_loss)

        if i_episode % print_every == 0:
            print("Episode {} completed, lasted {} turns -- Total Reward : {} -- Average DQN Loss : {}".format(i_episode, env.n_turns, ep_reward, ep_q_loss))

        # only save if optimisation has been done
        save_freq = 500# temp
        if i_episode % save_freq == 0 and policy_loss:
            saveStateDict(episode + i_episode, encoder, decoder, encoder_optimizer, decoder_optimizer, policy_loss, voc, encoder.embedding, save_dir)

        discriminator_active = False  # temp
        if discriminator_active:
            if i_episode % retrain_discriminator_every == 0:
                print('Updating Discriminator...')
                optimizer = torch.optim.Adam(env.AD.parameters(), lr=learning_rate)
                criterion = nn.CrossEntropyLoss()
                for i in range(1):
                    loss = trainAdversarialDiscriminatorOnLatestSeq2Seq(env.AD, policy, voc, ad_data, criterion, optimizer,
                                                                        embedding, 'data/save/Adversarial_Discriminator/', i)
                torch.save({
                    'iteration': i_episode,
                    'model': env.AD.state_dict(),
                    'opt': optimizer.state_dict(),
                    'loss': loss,
                    'voc_dict': voc.__dict__,
                    'embedding': embedding.state_dict()
                }, os.path.join(save_dir, '{}_{}_AD.tar'.format(i_episode, 'epochs')))


        # TODO: implement target/policy net (DDQN)?
        # if i_episode % TARGET_UPDATE == 0:
        #     target_net.load_state_dict(policy_net.state_dict())

    print('Training Complete!')
    return policy, env, total_rewards, dqn_losses


if __name__ == '__main__':
    policy, env, total_rewards, dqn_losses = train(num_episodes=1)
    chat(policy, env)