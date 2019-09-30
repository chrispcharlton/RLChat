from reinforcement_learning import train, chat

# train model using defaults (warm-started s2s, 50 eps)
policy, env,  total_rewards, dqn_losses = train(num_episodes=100)

# evaluate trained model
chat(policy, env)

import matplotlib.pyplot as plt

plt.plot(total_rewards)
plt.show()
plt.clf()
plt.plot(dqn_losses)
plt.show()
