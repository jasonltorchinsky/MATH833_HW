'''
Solves problem 2 of problem set 4.

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

    omega = 0.0
    sigma_u = 0.3

    d_gamma = 0.5
    gamma_hat = 0.0
    sigma_gamma = 0.2

    regime = 'i'
    

    helpStr = ("problem2.py --omega <omega parameter> "
               "--sigma_u <sigma_u parameter>  --d_gamma <d_gamma parameter> "
               "--gamma_hat <gamma_hat parameter>  "
               "--sigma_gamma <sigma_gamma parameter> ")
    try:
        # Dummy short names
        opts, args = getopt.getopt(argv, "hq:w:e:r:t:y:r:", ['omega=',
                                                           'sigma_u=',
                                                           'd_gamma=',
                                                           'gamma_hat=',
                                                           'sigma_gamma=',
                                                           'regime='])
    except getopt.GetoptError:
        print(helpStr)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(helpStr)
            sys.exit()
        elif opt in ('-q', '--omega'):
            omega = float(arg)
        elif opt in ('-w', '--sigma_u'):
            sigma_u = float(arg)
        elif opt in ('-e', '--d_gamma'):
            d_gamma = float(arg)
        elif opt in ('-r', '--gamma_hat'):
            gamma_hat = float(arg)
        elif opt in ('-t', '--sigma_gamma'):
            sigma_gamma = float(arg)
        elif opt in ('-r', '--regime'):
            regime = arg

    # Print startup message
    print(("Starting Problem 2 with:\n"
           "  omega       = {:.2f}.\n"
           "  sigma_u     = {:.2f}.\n"
           "  d_gamma     = {:.2f}.\n"
           "  gamma_hat   = {:.2f}.\n"
           "  sigma_gamma = {:.2f}.\n"
           "  regime      = " + regime + ".\n"
           ).format(omega, sigma_u, d_gamma, gamma_hat, sigma_gamma))

    
    ## Simulate trajectories of 100 time units
    rng = default_rng(1)
    
    dt = 0.05
    start_t = 0.0
    end_t = 100.0
    t = np.arange(start_t, end_t + dt, dt)
    nsteps = np.size(t)
    
    ntrials = 2000

    trials = np.zeros([ntrials, 3, nsteps]) # Trial,
                                            # state variable (u_1, u_2, gamma),
                                            # step

    # Initial condition
    trials[:, 0, 0] = 0.0
    trials[:, 1, 0] = 0.0
    trials[:, 2, 0] = 0.0

    for step in range(1, nsteps):
        for trial in range(ntrials) :
            trials[trial, :, step] = advance_state(rng, omega, sigma_u,
                                                   d_gamma, gamma_hat,
                                                   sigma_gamma, dt,
                                                   trials[trial, :, step-1])

    # Plot a single trial
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 6)

    gs = GridSpec(2, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, trials[0, 0, :], color='#E69F00', label=r'$u_1$')
    ax0.plot(t, trials[0, 1, :], color='#56B4E9', label=r'$u_2$')

    ax0.set_title('Single Trajectory')
    ax0.set_xlabel(r'$t$')
    ax0.legend()


    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, trials[0, 2, :], color='#009E73', label=r'$\gamma$')
    ax1.plot(t, 0*trials[0, 2, :], color='#000000')

    ax1.set_title('Single Trajectory')
    ax1.set_xlabel(r'$t$')
    ax1.legend()
    
    # Save plot to file
    plot_file_name = '2_traj_' + regime + '.png'
    plt.savefig(plot_file_name, dpi=300)

    # Calculate ensemble statistics
    num_u1_mean = np.mean(trials[:, 0, :], axis=0)
    num_u2_mean = np.mean(trials[:, 1, :], axis=0)

    num_u1_var = np.var(trials[:, 0, :], axis=0)
    num_u2_var = np.var(trials[:, 1, :], axis=0)
    num_u_cov  = np.zeros_like(t)
    for step in range(0, nsteps):
        num_u_cov[step] = np.cov(trials[:, :, step], rowvar=False)[0,1]

    # Plot ensemble statistics
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 6)

    gs = GridSpec(2, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, num_u1_mean, color='#E69F00', label=r'$\langle u_1 \rangle$')
    ax0.plot(t, num_u2_mean, color='#56B4E9', label=r'$\langle u_2 \rangle$')
    
    ax0.set_xlabel(r'$t$')
    ax0.set_title('Ensemble Means')
    ax0.legend()

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, num_u1_var, color='#E69F00', label=r'$Var(u_1)$')
    ax1.plot(t, num_u2_var, color='#56B4E9', label=r'$Var(u_2)$')
    ax1.plot(t, num_u_cov,  color='#009E73', label=r'$Cov(u_1,\ u_2)$')

    ax1.set_xlabel(r'$t$')
    ax1.set_title('Ensemble Variances')
    ax1.legend()

    # Save plot to file
    plot_file_name = '2_ens_stats_' + regime + '.png'
    plt.savefig(plot_file_name, dpi=300)
    
    # Plot equilibrium PDFs

    # Gaussian fits
    x = np.linspace(np.amin(trials[:,0:1,nsteps-1]-10),
                    np.amax(trials[:,0:1,nsteps-1]+10),
                    200)
    u1_fit = gauss(x, num_u1_mean[nsteps-1], num_u1_var[nsteps-1])
    u2_fit = gauss(x, num_u2_mean[nsteps-1], num_u2_var[nsteps-1])

    print(("Numerical statistical moments [u1] (excluding outliers):\n"
           "  mean       = {:.2f}.\n"
           "  variance     = {:.2f}.\n"
           "  skew     = {:.2f}.\n"
           "  kurtosis   = {:.2f}.\n"
           ).format(np.mean(remove_outliers(trials[:,0,nsteps-1]), axis=0),
                    np.var(remove_outliers(trials[:,0,nsteps-1]), axis=0),
                    stats.skew(remove_outliers(trials[:,0,nsteps-1]), axis=0),
                    stats.kurtosis(remove_outliers(trials[:,0,nsteps-1]), axis=0)
                    ))

    print(("Numerical statistical moments [u2] (excluding outliers):\n"
           "  mean       = {:.2f}.\n"
           "  variance     = {:.2f}.\n"
           "  skew     = {:.2f}.\n"
           "  kurtosis   = {:.2f}.\n"
           ).format(np.mean(remove_outliers(trials[:,1,nsteps-1]), axis=0),
                    np.var(remove_outliers(trials[:,1,nsteps-1]), axis=0),
                    stats.skew(remove_outliers(trials[:,1,nsteps-1]), axis=0),
                    stats.kurtosis(remove_outliers(trials[:,1,nsteps-1]), axis=0)
                    ))

    # Numerical
    nbins = 64
    [u1_hist, u1_bin_edges] = np.histogram(trials[:, 0, nsteps-1], nbins,
                                           density=True)
    u1_bin_centers = (u1_bin_edges[:nbins] + u1_bin_edges[1:]) / 2

    [u2_hist, u2_bin_edges] = np.histogram(trials[:, 1, nsteps-1], nbins,
                                           density=True)
    u2_bin_centers = (u2_bin_edges[:nbins] + u2_bin_edges[1:]) / 2

    ymax = 1.1*np.amax(np.array([np.amax(u1_fit),
                                 np.amax(u2_fit), 
                                 np.amax(u1_hist),
                                 np.amax(u2_hist)]))

    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 6)

    gs = GridSpec(1, 2, figure=fig)
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(u1_bin_centers, u1_hist, color='#E69F00', label=r'$u_1$')
    ax0.plot(x, u1_fit, color='#000000', linestyle='dashed')

    ax0.set_xlabel(r'$u_1$')
    ax0.set_ylabel('log(PDF)')
    if regime == 'iii':
        ax0.set_xlim([-2,2])
    else:
        ax0.set_xlim([-10,10])
    ax0.set_ylim([10**(-3), ymax])

    ax0.set_yscale('log')
    ax0.legend()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(u2_bin_centers, u2_hist, color='#56B4E9', label=r'$u_2$')
    ax1.plot(x, u2_fit, color='#000000', linestyle='dashed')
    
    
    ax1.set_xlabel(r'$u_2$')
    ax1.set_ylabel('log(PDF)')
    if regime == 'iii':
        ax1.set_xlim([-2,2])
    else:
        ax1.set_xlim([-10,10])
    ax1.set_ylim([10**(-3), ymax])

    ax1.set_yscale('log')
    ax1.legend()

    fig.suptitle('Equilbrium Distribution (t = {:.2f})'.format(end_t))

    # Save plot to file
    plot_file_name = '2_log_equ_pdf_' + regime + '.png'
    plt.savefig(plot_file_name, dpi=300)

    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 6)

    gs = GridSpec(1, 2, figure=fig)
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(u1_bin_centers, u1_hist, color='#E69F00', label=r'$u_1$')
    ax0.plot(x, u1_fit, color='#000000', linestyle='dashed')

    ax0.set_xlabel(r'$u_1$')
    ax0.set_ylabel('PDF')
    if regime == 'iii':
        ax0.set_xlim([-2,2])
    else:
        ax0.set_xlim([-10,10])
    ax0.set_ylim([0, ymax])

    ax0.legend()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(u2_bin_centers, u2_hist, color='#56B4E9', label=r'$u_2$')
    ax1.plot(x, u2_fit, color='#000000', linestyle='dashed')
    
    
    ax1.set_xlabel(r'$u_2$')
    ax1.set_ylabel('PDF')
    if regime == 'iii':
        ax1.set_xlim([-2,2])
    else:
        ax1.set_xlim([-10,10])
    ax1.set_ylim([0, ymax])

    ax1.legend()

    fig.suptitle('Equilbrium Distribution (t = {:.2f})'.format(end_t))

    # Save plot to file
    plot_file_name = '2_equ_pdf_' + regime + '.png'
    plt.savefig(plot_file_name, dpi=300)

def advance_state(rng, omega, sigma_u, d_gamma, gamma_hat,
                  sigma_gamma, dt, state):

    stoch = rng.standard_normal(np.size(state))
    
    state_out = np.zeros_like(state)
    state_out[0] = state[0] + (-state[2] * state[0] - omega * state[1]) * dt \
        + sigma_u / np.sqrt(2) * np.sqrt(dt) * stoch[0]
    state_out[1] = state[1] + (-state[2] * state[1] + omega * state[0]) * dt \
        + sigma_u / np.sqrt(2) * np.sqrt(dt) * stoch[1]
    state_out[2] = state[2] + (-d_gamma * (state[2] - gamma_hat)) * dt \
        + sigma_gamma * np.sqrt(dt) * stoch[2]
    
    return state_out

def gauss(x, mean, var):

    return 1 / np.sqrt(2 * np.pi * var) \
        * np.exp(-1 / (2 * var) * (x - mean) **2)

def remove_outliers(data, m=3.):

    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    if mdev:
        s = d/mdev
    else:
        s = 0.
    return data[s<m]

if __name__ == "__main__":
    main(sys.argv[1:])

