from rl_methods import *
from seq2seq.loader import loadModel, saveStateDict
from _config import save_every
from environment import Env

def train(load_dir='data\\save\\cb_model\\cornell movie-dialogs corpus\\2-2_500', save_dir="data\\rl_models\\DQNseq2seq", num_episodes=50, env=None):
    episode, encoder, decoder, encoder_optimizer, decoder_optimizer, policy, voc = loadModel(directory=load_dir)
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
            action = policy.select_action(state)
            reward, next_state, done = env.step(action)
            reward = torch.tensor([reward], device=device)
            action = env.state2tensors([action])[0]
            memory.push(state, action, next_state, reward, done)
            state = next_state
        print("Episode {} completed, lasted {} turns.".format(i_episode, length))
        loss = optimize_model(policy, memory, encoder_optimizer, decoder_optimizer)
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