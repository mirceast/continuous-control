import os
from ddpg_agent import Agent
from unityagents import UnityEnvironment
import numpy as np
from collections import deque

import torch
import matplotlib.pyplot as plt


import time

from misc import ddpg, find_state_mag


env = UnityEnvironment(
    file_name='./Reacher_Linux/Reacher.x86_64', no_graphics=True)

# states_mean, states_std = find_state_mag(env, max_t=300, n_episodes=300)
# fig, ax = plt.subplots()
# ax.bar(np.arange(len(states_mean)), states_mean, yerr=states_std)
# ax.set_xlabel("State features")
# fig.savefig("States magnitude.png")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

# Run DDPG
scores = ddpg(
    agent, env, folder="double_400_300_chunk_actor_1e-4_critic_1e-4_batch_128")

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')

plt.show()
