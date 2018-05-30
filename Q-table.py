import gym
import numpy as np
import random
from bokeh.plotting import figure, output_file, show
env = gym.make('FrozenLake-v0')

num_action = env.action_space.n
num_state = env.observation_space.n

q_table = np.zeros([num_state, num_action])

epsodes = 10000
max_step = 100
gamma = 0.95
min_explore_rate = 0.01
learn_rate = 0.8
decay_rate = 0.01
explore_rate = 1

#show explore rate data
rate_data = []
#show max_step in every episode
show_steps = []
for epsode in  range(epsodes):
    rate_data.append(explore_rate)
    state = env.reset()

    for step in range(max_step):
        real_random = random.uniform(0,1)
        if real_random < explore_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state,:])

        new_state, reward, done, info = env.step(action)
        q_table[state, action] = q_table[state][action] + \
                                 learn_rate * (gamma*np.max(q_table[new_state,:]) - q_table[state, action] + reward)
        if done:
            show_steps.append(step+1)
            break
        state = new_state
    if len(show_steps) <= epsode:
        show_steps.append(max_step)
    explore_rate -= (1 - min_explore_rate) / epsodes

print(q_table)


p = figure(title="the explore rate in every episode", width=800, height=800)
p.title.align = "center"
#plot the steps in every episode
p.line(y=show_steps, x=range(len(show_steps)), color= "navy", legend = "stpes")
#plot the change of export_rate
p.line(y=[item*100 for item in rate_data], x=range(len(rate_data)), color="red", legend = "explore rate")
p.xaxis.axis_label = "episode"
p.yaxis.axis_label = "explore rate: % / steps"

output_file("explore_rate.html")
show(p)


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

