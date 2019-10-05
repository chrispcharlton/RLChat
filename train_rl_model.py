from reinforcement_learning import train, chat

# train model using defaults (warm-started s2s, 50 eps)
policy, env,  total_rewards, dqn_losses = train(num_episodes=100)

# evaluate trained model
chat(policy, env)
