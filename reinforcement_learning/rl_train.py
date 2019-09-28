from reinforcement_learning.rl_methods import *
from seq2seq.loader import loadModel, saveStateDict
from reinforcement_learning.qnet import DQN
from _config import save_every, hidden_size, learning_rate
from reinforcement_learning.environment import Env

def train(load_dir='data\\save\\cb_model\\cornell movie-dialogs corpus\\2-2_500', save_dir="data\\rl_models\\DQNseq2seq", num_episodes=50, env=None):
    episode, encoder, decoder, encoder_optimizer, decoder_optimizer, voc = loadModel(directory=load_dir)
    searcher = RLGreedySearchDecoder(encoder, decoder, voc)
    embedding = nn.Embedding(voc.num_words, hidden_size)
    policy = DQN(hidden_size, embedding)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    memory = ReplayMemory(1000)
    env = env if env else Env(voc)

    # set episode number to 0 if starting from warm-started model. If loading rl-trained model continue from current number of eps
    if "\\rl_models\\" not in load_dir:
        episode = 0

    # RL training loop
    print("Training for {} episodes...".format(num_episodes))
    for i_episode in range(1, num_episodes+1):
        env.reset()
        state = env.state
        done = False
        length = 0
        while not done:
            length += 1
            action = searcher.select_action(state)
            reward, next_state, done = env.step(action)
            reward = torch.tensor([reward], device=device)
            action = env.state2tensors([action])[0]
            memory.push(state, action, next_state, reward, done)
            state = next_state
        print("Episode {} completed, lasted {} turns.".format(i_episode, length))
        loss = optimize_batch_q(policy, policy_optimizer, memory, encoder_optimizer, decoder_optimizer)
            # if done:
                # record episode duration?

        # only save if optimisation has been done
        if i_episode % save_every == 0 and loss:
            saveStateDict(episode + i_episode, encoder, decoder, encoder_optimizer, decoder_optimizer, loss, voc, encoder.embedding, save_dir)

        # TODO: implement target/policy net (DDQN)?
        # if i_episode % TARGET_UPDATE == 0:
        #     target_net.load_state_dict(policy_net.state_dict())

    return policy, env

if __name__ == '__main__':
    train(num_episodes=30)