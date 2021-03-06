'''
Solves problem 3 of problem set 4.

Author: Jason Torchinsky
Date: November 17th, 2021
'''

import numpy as np
from numpy.random import default_rng

import sys, getopt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy.stats as stats

def main(argv):

    a = 1
    f = 1
    sigma = 0.5

    g = 1
    sigma_o = 0.25
    

    helpStr = ("problem3.py -a <decay parameter> "
               "-f <deterministic forcing strength>  "
               "-s <stochastic forcing strength>")
    try:
        # Dummy short names
        opts, args = getopt.getopt(argv, "ha:f:s:g:z:", ['decay=',
                                                         'detforce=',
                                                         'stochforce=',
                                                         'obs=',
                                                         'obsnoise='])
    except getopt.GetoptError:
        print(helpStr)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(helpStr)
            sys.exit()
        elif opt in ('-a', '--decay'):
            a = float(arg)
        elif opt in ('-f', '--detforce'):
            f = float(arg)
        elif opt in ('-s', '--stochforce'):
            sigma = float(arg)
        elif opt in ('-g', '--obs'):
            g = float(arg)
        elif opt in ('-z', '--obsnoise'):
            sigma_o = float(arg)

    # Print startup message
    print(("Starting Problem 3 with:\n"
           "  a       = {:.2f}.\n"
           "  f       = {:.2f}.\n"
           "  sigma   = {:.2f}.\n"
           "  g       = {:.2f}.\n"
           "  sigma_o = {:.2f}.\n"
           ).format(a, f, sigma, g, sigma_o))

    
    ## Simulate trajectories of 100 time units
    rng = default_rng(1)
    
    dt = 0.01
    start_t = 0.0
    end_t = 100.0
    t = np.arange(start_t, end_t + dt, dt)
    nsteps = np.size(t)
    
    ntrials = 250

    trials = np.zeros([ntrials, 1, nsteps]) # Trial,
                                            # state variable (u),
                                            # step

    # Initial condition
    trials[:, 0, 0] = 0.0
    #trials[:, 0, 0] = f/a + (np.sqrt(sigma**2 / (2*a))
    #                         * rng.standard_normal([ntrials,]))
    obs = np.zeros_like(trials)
    
    for step in range(1, nsteps):
        for trial in range(ntrials) :
            trials[trial, :, step] = advance_state(rng, a, f, sigma, dt,
                                                   trials[trial, :, step-1])
            obs[trial, :, step] = observe_state(rng, g, sigma_o,
                                                trials[trial, :, step-1])

    # Plot a single trial
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 6)

    gs = GridSpec(2, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, trials[0, 0, :], color='#E69F00', label=r'$u$')
    
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, obs[0, 0, :] - trials[0, 0, :], color='#56B4E9',
             label=r'observed $u$ - true $u$')

    axs = [ax0, ax1]
    for ax in axs:
        ax.set_xlabel(r'$t$')
        ax.legend()
    
    fig.suptitle('Single Trajectory')
    

    
    # Save plot to file
    plot_file_name = '3a_traj.png'
    plt.savefig(plot_file_name, dpi=300)

    # Calculate ensemble statistics
    num_u_mean = np.mean(trials[:, 0, :], axis=0)
    num_u_var = np.var(trials[:, 0, :], axis=0)

    # Plot ensemble statistics
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 6)

    gs = GridSpec(2, 1, figure=fig)

    # Ensemble mean
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, num_u_mean, color='#E69F00', label=r'$\langle u \rangle$')
    ax0.plot(t, f/a * np.ones_like(t), color='#000000', linestyle='dashed')
    
    ax0.set_ylabel(r'$\langle r \rangle$')

    # Ensemble Variance
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, num_u_var, color='#E69F00', label=r'$Var(u)$')
    ax1.plot(t, sigma**2 / (2*a) * np.ones_like(t), color='#000000',
             linestyle='dashed')

    ax1.set_ylabel(r'$Var(u)$')
    
    # Elements common across plots
    axs = [ax0, ax1]
    for ax in axs:
        ax.set_xlabel(r'$t$')
        ax.legend()

    # Save plot to file
    plot_file_name = '3a_ens_stats.png'
    plt.savefig(plot_file_name, dpi=300)

def advance_state(rng, a, f, sigma, dt, state):

    stoch = rng.standard_normal(np.size(state))
    
    state_out = np.zeros_like(state)
    state_out = state + (-a * state + f) * dt \
        + sigma * np.sqrt(dt) * stoch[0]
    
    return state_out

def observe_state(rng, g, sigma_o, state):

    stoch = rng.standard_normal(np.size(state))
    
    return g * state + sigma_o * stoch[0]


if __name__ == "__main__":
    main(sys.argv[1:])

