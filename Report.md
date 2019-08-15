# Continuous control with DDPG

In this project I applied the [deep deterministic policy gradient](https://arxiv.org/abs/1509.02971) algorithm to the task of controlling a double-jointed arm for reaching and maintaining position close to a moving target. I have found that the most critical step for convergence is to run a number of update steps (here 10) only after a refractory period in which the networks are not updated (here 20 environment steps). I have also found that applying the [Double DQN trick](https://arxiv.org/abs/1509.06461) improved performance significantly, decreasing the training time by 37 percent.
<br><br>
### Methods
##### Environment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The target location is not given in the state vector - the agent must first find the location and then maintain position within a certain radius while the target is circling the agent with a constant angular velocity. The angular velocity changes each episode.

The task is considered solved when the agent gets an average score of +30 over 100 consecutive episodes.

The maximum number of steps for this environment is 1000, and this is what I have used in all the results below. It's worth noting that since the reward is cumulative, choosing a different (smaller) maximum number of steps will result in a smaller expected reward per episode.
<br><br>
##### DDPG
[DDPG](https://arxiv.org/abs/1509.02971) is basically [DQN](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf) for continuous action spaces. The problem with DQN is that it's doing a maximization over all possible discrete actions. This doesn't work in continuous spaces because the action space would have an infinite number of possible actions. This problem could be overcome to some extent by discretization; however, discretization is not optimal since it does not scale well and loses information. DDPG circumvents the problem by employing a different network, called an actor, to generate the best action given a certain state. Together with the state, this action is then taken as input by the critic, which outputs an expected return (Q value). Like DQN, DDPG then "forces" the Bellman equation to be true by adjusting the critic's weights such that the predicted return for the current (state, action) pair is as close as possible to the predicted return from the following (state, action) pair, as calculated by the Bellman equation. One difference between DDPG and DQN is that DDPG has a "soft update" for the target network, in which only a small proportion of the local network's weights are being copied.
The goal of the actor network is to learn the optimal action to take in each state. It does this by directly maximizing the critic's return estimate for the current (state, action) pair.
<br><br>
##### Double DDPG
[Double DQN](https://arxiv.org/abs/1509.06461) improves on the traditional DQN by decoupling action selection and action evaluation. Double DQN proposes to use the online network for choosing the next action, and evaluate that choice with the target network. This was shown to reduce the overoptimism of Q learning and to provide usually better performance. I thought of adapting this idea to DDPG by using the **local** actor network for selecting the next actions rather than the **target** actor network.
<br><br>
##### Actor network
For the actor network I used a feed-forward network with 3 fully-connected layers followed by ReLU activations. The first layer takes as input the 33-dimensional state. The last layer outputs a 4-dimensional action. The layer sizes were:
1. 33 -> 400
2. 400 -> 300
3. 300 -> 4

In most experiments I've also added batch normalization layers like so:
1. BN -> 33 -> 400 -> BN -> ReLU
2. 400 -> 300 -> BN -> ReLU
3. 300 -> 4
<br>
<br>

##### Critic network
The critic network is based on the same 3 fully-connected layers structure as the actor network. The first layer takes as input the 33-dimensional state. The last layer outputs a one-dimensional (state-action) estimated return this time. And the second layer is now 4 neurons larger because that's where the action given by the critic is added by concatenation. The layer sizes were:
1. 33 -> 400
2. 404 -> 300
3. 300 -> 1

In most experiments I've also added batch normalization layers like so:
1. BN -> 33 -> 400 -> BN -> ReLU
2. 404 -> 300
3. 300 -> 4
<br>
<br>

##### Hyperparameters and optimizers
I used the Adam optimizer with the default settings and no decay. The same seed was used for all experiments. The hyperparameters used are as follows:
- Replay buffer size: 1e6
- Batch size: 128 - 512
- Discount factor: 0.99
- Soft update mix: 1e-3
- Actor learning rate: 5e-5 - 5e-4
- Critic learning rate: 1e-4 - 1e-3
<br>
<br>

### Results
I first tried the algorithm as described - that is, updating the networks at each step. This did not go well, with the agent showing no signs of convergence. I then imposed a refractory period of 20 environment steps in which the agent would not learn. After this period is over, the agent samples from the buffer and learns 10 times. I call this approach "Chunks" in the plots below. I also changed some hyperparameters and tried with and without batch normalization layers.
<br><br>
<img src="Initial attempts.png">
<br><br>
It became immediately clear that the "learning in chunks" method is necessary. I also checked the possible absolute values of the state vector in order to decide whether a batch normalization layer on the input is necessary. I did this by running random actions for 300 time steps in 300 episodes.
<br><br>
<img src="States magnitude.png">
<br><br>
The large difference in magnitude was kind of expected - after all, different units are being thrown together in the state vector. But it's always nice to check. Anyway, it looked like a batch normalization layer on the input wouldn't be a bad idea.

Next I left the best initial attempt to train for longer. This corresponds to the batch size of 128, and a learning rate of 1e-4 for both networks. The DDPG agent solved the environments in 570 episodes, corresponding to one hour and 12 minutes on a 1080 Ti and a 7700K.
<br><br>
<img src="DDPG Solved.png">
<br><br>

Then I realized that the target actor is choosing the action for the critic - just like in vanilla DQN. Why not use the local network for choosing the action, as in Double DQN? This also makes some intuitive sense to me - after all, it's the local network who will actually choose the next action.
<br><br>
<img src="DDPG vs Double DDPG.png">
<br><br>

Double DDPG resulted in 37% shorter time necessary for the agent to solve the environment: only 359 episodes, or 44 minutes.
<br><br>
### Conclusion
In conclusion the regular DDPG as presented in the paper does not really work that well. Actually, it doesn't seem to work at all. It is critical that the actual learning is done in "chunks", by using a refractory period after which a number of learning steps are performed. This was certainly not something I expected after reading the paper, which makes no mention of such a procedure - the soft update should be the replacement. I have not played around with these two parameters, the refractory period and the number of training steps, but there could be potential to be gained by tweaking them.

I've tried the Double DQN trick, which is literally changing "target" to "local" in one line of code. This resulted in 37% reduction in training time, which is substantial.

### What I learnt
1. Don't blindly do what you find in the paper - adapt.
2. Always record your experiments. Never lose work by pressing ctrl+c.
3. DRL is super-sensitive to hyperparameter tuning. Use saved experiments to decide which way to go.