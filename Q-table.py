import gym
import numpy as np
import random

env = gym.make('FrozenLake-v0')

num_action = env.action_space.n
num_state = env.observation_space.n

q_table = np.zeros([num_state, num_action])

epsodes = 10000
max_step = 100
gamma = 0.95
min_explord_rate = 0.01
learn_rate = 0.8
decay_rate = 0.01
explord_rate = 1

for epsode in  range(epsodes):
    state = env.reset()

    for step in range(max_step):
        real_random = random.uniform(0,1)
        if real_random < explord_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state,:])

        new_state, reward, done, info = env.step(action)
        q_table[state, action] = q_table[state][action] + \
                                 learn_rate * (gamma*np.max(q_table[new_state,:]) - q_table[state, action] + reward)
        if done:
            break
        state = new_state
    explord_rate -= (1 - min_explord_rate) / epsodes

print(q_table)

all_reward = 0
for episode in range(100):
    state = env.reset()
    rewards = 0

    for _ in range(1000):
        #env.render()
        action = np.argmax(q_table[state,:])
        new_state, reward, done, info = env.step(action)
        rewards += reward
        if done:
            break
        state = new_state
    all_reward += rewards
print("the all reward in 100 episodes is {}".format(all_reward))

