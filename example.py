from frozen_lakes import ModifiedFrozenLake, MAPS
from algorithm import solver
from vis import plot_dist
import numpy as np

MAP_NAME = '5x4uturn'
desc = np.array(MAPS[MAP_NAME], dtype='c')
env = ModifiedFrozenLake(map_name=MAP_NAME, cyclic_mode=True)
# Later; assert env timer is not exceeded by max collection steps
beta = 5
max_iter = 10
pi0 = np.ones((env.nS, env.nA)) / env.nA

free_energy = solver(env, prior_policy=pi0, beta=beta,
                     max_iter=max_iter, verbose=True, max_collection_steps=300,
                     num_trajectories=30, bulk_window=100, sa_window=50)

print(free_energy)
Q_sa = -free_energy
optimal_policy = np.exp(beta*Q_sa)
optimal_policy = optimal_policy / np.sum(optimal_policy, axis=1, keepdims=True)

greedy_actions = np.argmax(optimal_policy, axis=1)
greedy_policy = np.zeros_like(optimal_policy)
greedy_policy[np.arange(greedy_policy.shape[0]), greedy_actions] = 1

# calculate v(s) from free energy
v = 1/beta * np.log(np.sum(np.exp(beta * Q_sa) * pi0, axis=1))

plot_dist(desc, [optimal_policy, v], [greedy_policy, v], force_colorbar=True,
          titles=['Extracted policy', 'Greedy policy'], filename=f'figures/{MAP_NAME}_optimal_policy.png')

# tester(env, prior_policy=None, beta=beta)
