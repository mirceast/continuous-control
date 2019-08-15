from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import torch

from ddpg_agent import Agent

env = UnityEnvironment(
    file_name='./Reacher_Linux/Reacher.x86_64', no_graphics=False)


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

agent.actor_local.load_state_dict(
    torch.load("checkpoint_actor_Double_DDPG.pth"))
agent.critic_local.load_state_dict(
    torch.load("checkpoint_critic_Double_DDPG.pth"))

n_episodes = 20
max_t = 200

for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    state = env_info.vector_observations[0]
    for t in range(max_t):
        actions = agent.act(state)
        env_info = env.step(actions)[brain_name]
        state = env_info.vector_observations[0]
        done = env_info.local_done[0]
        if done:
            break
