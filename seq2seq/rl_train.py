from seq2seq.rl_methods import *
from seq2seq.loader import loadModel, saveStateDict
from seq2seq._config import save_every
from seq2seq.environment import Env

# directory for warm-started seq2seq model
load_dir = 'data\\save\\cb_model\\cornell movie-dialogs corpus\\2-2_500'
save_dir = "data\\rl_models\\DQNseq2seq"

episode, encoder, decoder, encoder_optimizer, decoder_optimizer, searcher, voc = loadModel(directory=load_dir)
memory = ReplayMemory(1000)
env = Env(voc)

# RL training loop
num_episodes = 50
for i_episode in range(1, num_episodes+1):
    print("Episode", i_episode)
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
    loss = optimize_model(searcher, memory, encoder_optimizer, decoder_optimizer)
        # if done:
            # record episode duration?

    # only save if optimisation has been done
    if i_episode % save_every == 0 and loss:
        saveStateDict(episode + i_episode, encoder, decoder, encoder_optimizer, decoder_optimizer, loss, voc, encoder.embedding, save_dir)

    # TODO: implement target/policy net (DDQN)?
    # if i_episode % TARGET_UPDATE == 0:
    #     target_net.load_state_dict(policy_net.state_dict())

