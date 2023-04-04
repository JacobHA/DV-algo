import numpy as np
import matplotlib.pyplot as plt

gamma = 1


def solver(env, prior_policy=None, beta=1, max_iter=100,
           max_collection_steps=1000, num_trajectories=10, bulk_window=100, sa_window=None, verbose=False):
    """
    Solves the given environment using the value iteration algorithm.
    :param env: The environment to solve.
    :param prior_policy: The prior policy to use. If None, the uniform policy is used.
    :param beta: The inverse temperature parameter.
    :param max_iter: The maximum number of iterations to run.
    :param max_collection_steps: The maximum number of steps to take when collecting
    trajectories.
    :param verbose: Whether to print the current iteration number.
    :return: The free energy of the optimal policy.
    """

    if sa_window is None:
        sa_window = bulk_window
    # Initialize the free energy.
    free_energy = np.zeros((env.nS, env.nA))

    # Initialize the prior policy.
    if prior_policy is None:
        prior_policy = np.ones((env.nS, env.nA)) / env.nA

    # For all states and actions, calculate the energy of a trajectory
    # starting in that state and taking that action.
    energy = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        for a in range(env.nA):
            if s == 0 and a == 0:
                traj_energy = perstep_energy(env, s, a, prior_policy,
                                             max_collection_steps=5*max_collection_steps,
                                             num_trajectories=2*num_trajectories)
            else:
                traj_energy = perstep_energy(env, s, a, prior_policy,
                                             max_collection_steps=max_collection_steps,
                                             num_trajectories=num_trajectories)
            make = False
            global_eps = None
            if s == 0:  # and a == 0:
                make = False  # True

            if s == 0 and a == 0:
                epsilon, energy[s, a] = extract_sa_dependence2(
                    traj_energy, bulk=None, bulk_window=bulk_window, sa_window=sa_window,
                    make_plots=make, plot_title=f'({s}, {a})')
                global_eps = epsilon

            epsilon, energy[s, a] = extract_sa_dependence2(
                traj_energy, bulk=global_eps, bulk_window=bulk_window, sa_window=sa_window,
                make_plots=make, plot_title=f'({s}, {a})')

            # print(epsilon)
    # print(energy)
    free_energy = energy
    for i in range(max_iter):
        # Print the current iteration number.
        if verbose:
            print(f'Iteration {i}')
        # Extract a policy from the free energy.
        F = free_energy - np.mean(free_energy, axis=1, keepdims=True)
        # print(F)
        policy = np.exp(-beta * F)
        policy = policy / np.sum(policy, axis=1, keepdims=True)
        # print(policy)

        # Run the policy, calculating the KL divergence
        # between the prior and the current policy.
        kl_divergence = np.zeros((env.nS, env.nA))
        for s in range(env.nS):
            for a in range(env.nA):
                traj_kl = trajectory_kld(env, s, a, policy, prior_policy,
                                         max_collection_steps=max_collection_steps,
                                         num_trajectories=num_trajectories)
                _, kl_divergence[s, a] = extract_sa_dependence(
                    traj_kl, bulk_window=bulk_window)
        # kl_divergence /= max_collection_steps
        # Use the formula E = F + 1/beta KL to update the free energy.
        # print(kl_divergence)
        free_energy = energy - 1 / beta * kl_divergence
    return free_energy


def perstep_energy(env, start_s, start_a, policy, max_collection_steps=1000, num_trajectories=10):
    """
    Runs trajectories in the given environment, using the given policy.
    :param env: The environment to run the trajectories in.
    :param policy: The policy to use.
    :param max_steps: The maximum number of steps to take.
    :return: The total reward obtained.
    """
    energy = np.zeros((num_trajectories, max_collection_steps+1))
    for traj in range(num_trajectories):
        # Reset the environment.
        env.reset()

        # Set the state to start_s.
        env.s = start_s

        # Take a maximum number of steps
        for t in range(1, max_collection_steps + 1):
            s = env.s
            # Take the action.
            if t == 0:
                a = start_a
            else:
                # Draw from the prior policy.
                a = np.random.choice(env.nA, p=policy[s])

            s_next, r, done, _ = env.step(a)

            # Update the energy.
            energy[traj][t] = -gamma**t * r
            # If the episode is done, break.
            if done:
                break

            # Update the state.
            s = s_next

    return energy.mean(axis=0)


def trajectory_energy(env, start_s, start_a, policy, max_collection_steps=1000, num_trajectories=10):
    """
    Runs trajectories in the given environment, using the given policy.
    :param env: The environment to run the trajectories in.
    :param policy: The policy to use.
    :param max_steps: The maximum number of steps to take.
    :return: The total reward obtained.
    """
    energy = np.zeros((num_trajectories, max_collection_steps+1))
    for traj in range(num_trajectories):
        # Reset the environment.
        env.reset()

        # Set the state to start_s.
        env.s = start_s

        # Take a maximum number of steps
        for t in range(1, max_collection_steps + 1):
            s = env.s
            # Take the action.
            if t == 0:
                a = start_a
            else:
                # Draw from the prior policy.
                a = np.random.choice(env.nA, p=policy[s])

            s_next, r, done, _ = env.step(a)

            # Update the energy.
            energy[traj][t] = energy[traj][t - 1] + -gamma**t * r
            # If the episode is done, break.
            if done:
                break

            # Update the state.
            s = s_next

    return energy.mean(axis=0)


def trajectory_kld(env, start_s, start_a, policy, prior_policy, max_collection_steps=1000, num_trajectories=10):
    """
    Runs trajectories in the given environment, using the given policy.
    :param env: The environment to run the trajectories in.
    :param policy: The policy to use.
    :param max_steps: The maximum number of steps to take.
    :return: The total reward obtained.
    """
    kld = np.zeros((num_trajectories, max_collection_steps))
    for traj in range(num_trajectories):
        # Reset the environment.
        env.reset()

        # Set the state to start_s.
        env.s = start_s
        s = env.s

        # Take a maximum number of steps
        for t in range(max_collection_steps):
            a = None
            # Take the action.
            if t == 0:
                a = start_a

            else:
                # Draw from the prior policy.
                a = np.random.choice(env.nA, p=prior_policy[s])

            s_next, r, done, _ = env.step(a)

            # Update the energy.
            kld[traj, t] = kld[traj, t-1] + gamma**t * prior_policy[s, a] *\
                np.log(prior_policy[s, a] / policy[s, a])
            # * prior_policy[s, a]

            # If the episode is done, break.
            if done:
                break

            # Update the state.
            s = s_next

    return kld.mean(axis=0)


def perstep_kld(env, start_s, start_a, policy, prior_policy, max_collection_steps=1000, num_trajectories=10):
    """
    Runs trajectories in the given environment, using the given policy.
    :param env: The environment to run the trajectories in.
    :param policy: The policy to use.
    :param max_steps: The maximum number of steps to take.
    :return: The total reward obtained.
    """
    kld = np.zeros((num_trajectories, max_collection_steps))
    for traj in range(num_trajectories):
        # Reset the environment.
        env.reset()

        # Set the state to start_s.
        env.s = start_s
        s = env.s

        # Take a maximum number of steps
        for t in range(max_collection_steps):
            a = None
            # Take the action.
            if t == 0:
                a = start_a

            else:
                # Draw from the prior policy.
                a = np.random.choice(env.nA, p=prior_policy[s])

            s_next, r, done, _ = env.step(a)

            # Update the energy.
            kld[traj, t] = gamma**t * prior_policy[s, a] *\
                np.log(prior_policy[s, a] / policy[s, a])

            # If the episode is done, break.
            if done:
                break

            # Update the state.
            s = s_next

    return kld.mean(axis=0)


def extract_sa_dependence(values_over_time, bulk=None, bulk_window=100, sa_window=50, make_plots=False, plot_title=None):
    vals_by_N = values_over_time[1:] / np.arange(1, len(values_over_time))
    if bulk is None:
        bulk = np.mean(vals_by_N[-bulk_window:])
    if make_plots:
        # Plot the vals by N, with a horizontal line at the bulk level,
        # and a vertical line at the state-action window.
        plt.figure()
        plt.title(f'Values/N: {plot_title}')
        plt.plot(vals_by_N, color='b')
        plt.axhline(bulk, color='k')
        plt.axvline(len(values_over_time) - sa_window,
                    color='r', label='State-action window')
        plt.xlabel(r'Bulk window $\to$ max steps')
        plt.xlim(len(values_over_time) - bulk_window, len(values_over_time))
        # Set the ylim based on values in the window.
        plt.ylim(np.min(vals_by_N[-bulk_window:]),
                 np.max(vals_by_N[-bulk_window:]))
        plt.legend()
        plt.savefig(f'figures/zoom_vals_by_N_{plot_title}.png')
        plt.close()

        plt.figure()
        plt.title(f'Values/N: {plot_title}')
        plt.plot(vals_by_N, color='b')
        plt.xlim(0, len(values_over_time))
        plt.axhline(bulk, color='k')
        plt.axvline(len(values_over_time) - sa_window,
                    color='r', label='State-action window')
        plt.axvline(len(values_over_time) - bulk_window,
                    color='g', label='Bulk window')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'figures/vals_by_N_{plot_title}.png')
        plt.close()

    # Compute the state-action dependence.
    w_sa = values_over_time - np.arange(0, len(values_over_time)) * bulk
    sa_dep = np.mean(w_sa[-sa_window:])
    if make_plots:
        # Plot the values over time, with a line with slope bulk, intersecting
        # the last data point.
        plt.figure()
        plt.title(f'Values over time: {plot_title}')
        plt.plot(w_sa, color='b')
        plt.axhline(sa_dep, color='k', label='State-action dependence')
        plt.axvline(len(values_over_time) - sa_window, color='r',
                    label='State-action window')
        # plt.plot()
        # plt.xlim(len(values_over_time) - sa_window, len(values_over_time))
        # plt.ylim(np.min(values_over_time[-sa_window:]),
        #          np.max(values_over_time[-sa_window:]))
        # # Plot a line with slope bulk and intercept sa_dep
        # plt.plot([N_0, len(values_over_time)],
        #          [bulk * N_0 + sa_dep, bulk * len(values_over_time) + sa_dep],
        #          color='k', label=f'sa dep: {sa_dep}')
        plt.legend()
        plt.savefig(f'figures/zoom_vals_over_time_{plot_title}.png')
        plt.close()

    return bulk, sa_dep


def extract_sa_dependence2(per_step_over_time, bulk=None, bulk_window=100, sa_window=50, make_plots=False, plot_title=None):
    if bulk is None:
        bulk = np.mean(per_step_over_time[-bulk_window:])
    if make_plots:
        # Plot the vals by N, with a horizontal line at the bulk level,
        # and a vertical line at the state-action window.
        plt.figure()
        plt.title(f'Values/N: {plot_title}')
        plt.plot(per_step_over_time, color='b')
        plt.axhline(bulk, color='k')
        plt.axvline(len(per_step_over_time) - sa_window,
                    color='r', label='State-action window')
        plt.xlabel(r'Bulk window $\to$ max steps')
        plt.xlim(len(per_step_over_time) -
                 bulk_window, len(per_step_over_time))
        # Set the ylim based on values in the window.
        plt.ylim(np.min(per_step_over_time[-bulk_window:]),
                 np.max(per_step_over_time[-bulk_window:]))
        plt.legend()
        plt.savefig(f'figures/zoom_vals_by_N_{plot_title}.png')
        plt.close()

        plt.figure()
        plt.title(f'Values/N: {plot_title}')
        plt.plot(per_step_over_time, color='b')
        plt.xlim(0, len(per_step_over_time))
        plt.axhline(bulk, color='k')
        plt.axvline(len(per_step_over_time) - sa_window,
                    color='r', label='State-action window')
        plt.axvline(len(per_step_over_time) - bulk_window,
                    color='g', label='Bulk window')
        plt.yscale('log')
        plt.legend()
        plt.savefig(f'figures/vals_by_N_{plot_title}.png')
        plt.close()

    # Compute the state-action dependence.
    # Sum all the values per step
    cumulative_values = np.cumsum(per_step_over_time)
    total_value = np.sum(per_step_over_time)
    # sa_dep = total_value - len(per_step_over_time) * bulk
    w_sa = cumulative_values - np.arange(0, len(per_step_over_time)) * bulk
    sa_dep = np.mean(w_sa[-sa_window:])
    if make_plots:
        # Plot the cumulative values over time, with a line at bulk
        plt.figure()
    #     plt.title(f'Values over time: {plot_title}')
        plt.plot(w_sa, color='b')
        # # Plot a line with slope bulk and intercept sa_dep
        # plt.plot([0, len(per_step_over_time)],
        #          [bulk * 0, bulk * len(per_step_over_time)],
        #          color='k', label=f'sa dep: {sa_dep}')
        plt.axhline(sa_dep, color='k', label='State-action dependence')
        plt.legend()
        plt.savefig(f'figures/zoom_vals_over_time_{plot_title}.png')
        plt.close()

    return bulk, sa_dep


# def tester(env, prior_policy=None, beta=1, max_collection_steps=100):
#     if prior_policy is None:
#         prior_policy = np.ones((env.nS, env.nA)) / env.nA
#     energies = trajectory_energy(env, 0, 0, prior_policy,
#                                  max_collection_steps=max_collection_steps, num_trajectories=10)

#     import matplotlib.pyplot as plt
#     plt.figure()
#     E_by_N = energies / np.arange(1, len(energies) + 1)
#     plt.plot(E_by_N)
#     plt.xlabel('Steps in Trajectory')
#     plt.ylabel('Accumulated Energy / Number of Steps')
#     plt.title('Average Energy Throughout Trajectory')
#     plt.savefig('energy.png')

#     epsilon = np.mean(E_by_N[-30:])
#     print(f'Epsilon: {epsilon}')
#     e_sa = np.mean(E_by_N[-5:] - epsilon)
#     print(f'e_sa: {e_sa}')
