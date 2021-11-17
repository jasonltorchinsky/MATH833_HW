'''
Solves problem 1 of problem set 4.

Author: Jason Torchinsky
Date: November 15th, 2021
'''

import numpy as np
from numpy.random import default_rng

import sys, getopt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy.stats as stats

def main(argv):

    lmbda = 1.0
    b = -1.0
    c = 1.0
    

    helpStr = ("problem1.py --lambda <lmbda parameter> --b <b parameter> "
               "--c <c parameter>")
    try:
        # Dummy short names
        opts, args = getopt.getopt(argv, "hl:b:c:", ['lambda=',
                                                     'b=',
                                                     'c='])
    except getopt.GetoptError:
        print(helpStr)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(helpStr)
            sys.exit()
        elif opt in ('-l', '--lambda'):
            lmbda = float(arg)
        elif opt in ('-b', '--b'):
            b = float(arg)
        elif opt in ('-c', '--c'):
            c = float(arg)

    # Print startup message
    print(("Starting Problem 1 with:\n"
           "  lambda = {:.2f}.\n"
           "  b      = {:.2f}.\n"
           "  c      = {:.2f}.\n"
           ).format(lmbda, b, c))

    ## Simulate trajectories of 100 time units
    rng = default_rng(1)
    
    dt = 0.05
    start_t = 0.0
    end_t = 100.0
    t = np.arange(start_t, end_t + dt, dt)
    nsteps = np.size(t)
    
    ntrials = 1000

    trials = np.zeros([ntrials, 1, nsteps]) # Trial, state variable (x),
                                            # step

    # Initial condition
    trials[:, 0, 0] = (b + c) / 3

    for step in range(1, nsteps):
        for trial in range(ntrials) :
            trials[trial, :, step] = advance_state(rng, lmbda, b, c, dt,
                                                   trials[trial, :, step-1])

    # Plot a single trial
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 3)

    ax = fig.add_subplot()
    ax.plot(t, trials[0, 0, :], color='#000000')

    ax.set_title('Single Trajectory')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x(t)$')
    ax.legend()

    # Save plot to file
    plot_file_name = '1_traj.png'
    plt.savefig(plot_file_name, dpi=300)

    # Calculate ensemble statistics
    num_mean = np.mean(trials[:, 0, :], axis=0)
    num_var = np.var(trials[:, 0, :], axis=0)

    true_mean = (b + c) / 2 * np.ones_like(num_mean)
    true_var = (c - b)**2 / 12 * np.ones_like(num_mean)

    # Plot ensemble statistics
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 6)

    gs = GridSpec(3, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, num_mean, color='#009E73', label='Numerical')
    ax0.plot(t, true_mean, color='#000000', linestyle='dashed', label='True')
    
    ax0.set_xlabel(r'$t$')
    ax0.set_ylabel(r'$\left\langle x(t) \right\rangle$')
    ax0.set_title('Ensemble Mean')
    ax0.legend()

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, num_var, color='#009E73', label='Numerical')
    ax1.plot(t, true_var, color='#000000', linestyle='dashed', label='True')

    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel(r'$Var\left( x(t) \right)$')
    ax1.set_title('Ensemble Variance')
    ax1.legend()

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.hist(trials[:, :, nsteps-1], 30, color='#009E73', density=True)
    ax2.plot(np.array([b,c]), np.array([1,1])/(c-b), color='#000000',
             linestyle='dashed')

    ax2.set_xlabel(r'$x(t)$')
    ax2.set_title('Equilbrium PDF')

    # Save plot to file
    plot_file_name = '1_ens_stats.png'
    plt.savefig(plot_file_name, dpi=300)

def advance_state(rng, lmbda, b, c, dt, state):
    
    stoch = rng.standard_normal(np.size(state))
    
    state_out = state + -lmbda * (state - (b + c)/2) * dt \
        + np.sqrt(dt) * np.sqrt(lmbda * (state - b) * (c - state)) * stoch
    
    if state_out < b:
        state_out = b
    elif state_out > c:
        state_out = c
    
    return state_out

if __name__ == "__main__":
    main(sys.argv[1:])

