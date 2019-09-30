from reinforcement_learning.rl_methods import *
from seq2seq.loader import loadModel, saveStateDict
from reinforcement_learning.qnet import DQN
from _config import save_every, hidden_size, learning_rate
from reinforcement_learning.environment import Env

def train(load_dir='data\\save\\cb_model\\cornell movie-dialogs corpus\\2-2_500', save_dir="data\\rl_models\\DQNseq2seq", num_episodes=50, env=None):
    episode, encoder, decoder, encoder_optimizer, decoder_optimizer, voc = loadModel(directory=load_dir)
    policy = RLGreedySearchDecoder(encoder, decoder, voc)
    embedding = nn.Embedding(voc.num_words, hidden_size)
    qnet = DQN(hidden_size, embedding)
    qnet_optimizer = torch.optim.Adam(qnet.parameters(), lr=learning_rate)
    memory = ReplayMemory(1000)
    env = env if env else Env(voc)

    # set episode number to 0 if starting from warm-started model. If loading rl-trained model continue from current number of eps
    if "\\rl_models\\" not in load_dir:
        episode = 0

    total_rewards = []
    dqn_losses = []

    # RL training loop
    print("Training for {} episodes...".format(num_episodes))
    for i_episode in range(1, num_episodes+1):
        env.reset()
        state = env.state
        done = False
        length = 0
        total_reward = 0
        total_q = 0
        while not done:
            length += 1
            action, prob = policy(state)
            prob = torch.tensor([torch.mean(prob)])
            reward, next_state, done = env.step(action)
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            memory.push(state, action, next_state, reward, done, prob)
            state = next_state
        print("Episode {} completed, lasted {} turns.".format(i_episode, env.n_turns))
        dqn_loss, policy_loss = optimize_batch_q(policy, qnet, qnet_optimizer, memory, encoder_optimizer, decoder_optimizer)
        total_rewards.append(total_reward)
        dqn_loss = dqn_loss if dqn_loss is not None else 0
        dqn_losses.append(dqn_loss)

        # only save if optimisation has been done
        if i_episode % save_every == 0 and policy_loss:
            saveStateDict(episode + i_episode, encoder, decoder, encoder_optimizer, decoder_optimizer, policy_loss, voc, encoder.embedding, save_dir)

        # TODO: implement target/policy net (DDQN)?
        # if i_episode % TARGET_UPDATE == 0:
        #     target_net.load_state_dict(policy_net.state_dict())

    return policy, env, total_rewards, dqn_losses

if __name__ == '__main__':
    train(num_episodes=30)