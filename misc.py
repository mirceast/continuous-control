import os
import numpy as np
import time
from collections import deque
import glob

import matplotlib.pyplot as plt
import torch


def ddpg(agent, env, n_episodes=1000, max_t=1000, scores_window=100, progress_every=2, save_every=60, folder=None):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    scores_deque = deque(maxlen=scores_window)
    scores = []
    mean_scores = []
    t_start = time.time()
    best_score = -np.inf

    progress_t = time.time()
    saved_t = time.time()
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        agent.reset()
        score = 0

        t_episode = time.time()
        for t in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
            if progress_every > 0 and time.time() - progress_t >= progress_every:
                print('\rAverage score: {:.2f}\tTime: {}'.format(
                    np.mean(scores_deque), seconds_to_time_str(time.time() - t_start)), end="")
                progress_t = time.time()
            if save_every > 0 and time.time() - saved_t >= save_every:
                saved_t = time.time()
                save_agent(agent, scores=scores, mean_scores=mean_scores, agent_name='',
                           train_time=seconds_to_time_str(time.time() - t_start), folder=folder)

        scores_deque.append(score)
        scores.append(score)
        mean_scores.append(np.mean(scores_deque))

        if np.mean(scores_deque) >= 30:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_deque)))
            save_agent(agent, scores=scores, mean_scores=mean_scores, agent_name='SOLVED',
                       train_time=seconds_to_time_str(time.time() - t_start), folder=folder)
            break

        if np.mean(scores_deque) > best_score:
            best_score = np.mean(scores_deque)
            save_agent(agent, scores=scores, mean_scores=mean_scores, agent_name='',
                       train_time=seconds_to_time_str(time.time() - t_start), folder=folder)

    return scores


def find_state_mag(env, max_t=1000, n_episodes=1000):
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size
    states = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        state = env_info.vector_observations[0]
        for t in range(max_t):
            states.append(state)
            actions = np.random.randn(num_agents, action_size)
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            state = env_info.vector_observations[0]
            done = env_info.local_done[0]
            if done:
                break
    states = np.array(states)
    states = np.abs(states)
    return np.mean(states, axis=0), np.std(states, axis=0)


def seconds_to_time_str(t):
    if t < 0:
        raise Exception("Negative time?")
    if t < 60:
        return "{:02d} seconds".format(int(t))
    elif t >= 60 and t < 3600:
        return "{:04.1f} minutes".format(t/60)
    elif t >= 3600:
        return "{:04.1f} hours".format(t/3600)


def save_agent(agent, scores=None, mean_scores=None, agent_name='', train_time='', folder=None):
    # Make sure save folder exists
    if folder is None:
        folder = os.getcwd()
    if not os.path.isdir(folder):
        os.makedirs(folder)
    # Figure out the root of the resulting file names
    if agent_name != "":
        name = "agent_" + agent_name + "_"
    else:
        name = "agent_"

    if train_time != "":
        name = name + "train_time_" + train_time.replace(" ", "_") + "_"

    # Save agent weights
    save_path = os.path.join(folder, name + "checkpoint_actor.pth")
    torch.save(agent.actor_local.state_dict(), save_path)
    save_path = os.path.join(folder, name + "checkpoint_critic.pth")
    torch.save(agent.critic_local.state_dict(), save_path)

    # Save agent scores
    if scores is not None:
        save_path = os.path.join(folder, name + "scores.np")
        np.savetxt(save_path, scores)
    if mean_scores is not None:
        save_path = os.path.join(folder, name + "mean_scores.np")
        np.savetxt(save_path, mean_scores)


def load_agent(agent_name="", train_time="last", folder=None):
    if folder is None:
        folder = os.getcwd()
    if agent_name != "":
        name = "agent_" + agent_name + "_"
    else:
        name = "agent_"
    if train_time != "last":
        name = name + "train_time_" + train_time.replace(" ", "_") + "_"
    else:
        files = glob.glob(os.path.join(folder, "agent*mean_scores.np"))
        files.sort(key=os.path.getmtime)
        files = files[-1]
        files = os.path.split(files)[1]
        name = files.split("mean_scores")[0]
    path_scores = os.path.join(folder, name + "scores.np")
    path_mean_scores = path_scores.replace("_scores", "_mean_scores")
    scores = np.loadtxt(path_scores)
    mean_scores = np.loadtxt(path_mean_scores)

    actor_dict = torch.load(os.path.join(
        folder, name + "checkpoint_actor.pth"))
    critic_dict = torch.load(os.path.join(
        folder, name + "checkpoint_critic.pth"))

    return scores, mean_scores, actor_dict, critic_dict


def load_folders(folders, train_time="last"):
    scores = []
    mean_scores = []
    actor_dicts = []
    critic_dicts = []
    for i in range(len(folders)):
        score, mean_score, actor_dict, critic_dict = load_agent(
            train_time=train_time, folder=folders[i])
        scores.append(score)
        mean_scores.append(mean_score)
        actor_dicts.append(actor_dict)
        critic_dicts.append(critic_dict)
    return mean_scores, scores, actor_dicts, critic_dicts


def show_plots(mean_scores, scores, labels=None, max_episodes=None, only_mean=False, legend_outside=False):
    if max_episodes == None:
        # Find max number of episodes
        max_episodes = 0
        for i in range(len(mean_scores)):
            if len(mean_scores[i]) > max_episodes:
                max_episodes = len(mean_scores[i])

    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap("jet", max([len(mean_scores), 2]))
    for i in range(len(mean_scores)):
        if labels is not None:
            label = labels[i]
        else:
            label = None
        mean_score = mean_scores[i]
        score = scores[i]
        if len(mean_score) < max_episodes:
            mean_score = np.concatenate(
                (mean_score, np.nan * np.ones(max_episodes-len(mean_score))))
            score = np.concatenate(
                (score, np.nan * np.ones(max_episodes-len(score))))
        if not only_mean:
            ax.plot(np.arange(1, max_episodes+1),
                    score, alpha=0.3, color=cmap(i))
        ax.plot(np.arange(1, max_episodes+1), mean_score,
                label=label, color=cmap(i), linewidth=2)
    if labels is not None:
        if legend_outside:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.legend()
    ax.set_xlabel("# episodes")
    ax.grid()
