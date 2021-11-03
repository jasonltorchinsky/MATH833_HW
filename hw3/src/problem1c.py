'''
Solves problem 1c of problem set 3.

Author: Jason Torchinsky
Date: November 3rd, 2021
'''

import numpy as np
from numpy.random import default_rng

import sys, getopt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy.stats as stats

def main(argv):

    F_11 = -1.0
    F_12 = 0.0
    F_21 = 0.0
    F_22 = -1.0
    sigma_1 = 1.0
    sigma_2 = 1.0
    g_1 = 1.0
    g_2 = 1.0
    sigma_1o = 1.0
    sigma_2o = 1.0

    helpStr = ("problem1.py --F_11 <F_11 parameter> --F_12 <F_12 parameter> "
               "--F_21 <F_21 parameter> --F_22 <F_22 parameter> "
               "--sigma_1 <sigma_1 parameter>  --sigma_2 <sigma_2 parameter>")
    try:
        # Dummy short names
        opts, args = getopt.getopt(argv, "hq:w:e:r:t:y:u:i:o:p:", ['F_11=',
                                                           'F_12=',
                                                           'F_21=',
                                                           'F_22=',
                                                           'sigma_1=',
                                                           'sigma_2=',
                                                           'g_1=',
                                                           'g_2=',
                                                           'sigma_1o=',
                                                           'sigma_2o='])
    except getopt.GetoptError:
        print(helpStr)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(helpStr)
            sys.exit()
        elif opt in ('-q', '--F_11'):
            F_11 = float(arg)
        elif opt in ('-w', '--F_12'):
            F_12 = float(arg)
        elif opt in ('-e', '--F_21'):
            F_21 = float(arg)
        elif opt in ('-r', '--F_22'):
            F_22 = float(arg)
        elif opt in ('-t', '--sigma_1'):
            sigma_1 = float(arg)
        elif opt in ('-y', '--sigma_2'):
            sigma_2 = float(arg)
        elif opt in ('-u', '--g_1'):
            g_1 = float(arg)
        elif opt in ('-i', '--g_2'):
            g_2 = float(arg)
        elif opt in ('-o', '--sigma_1o'):
            sigma_1o = float(arg)
        elif opt in ('-o', '--sigma_2o'):
            sigma_2o = float(arg)

    # Print startup message
    print(("Starting Problem 1 with:\n"
           "  F_11     = {:.2f}.\n"
           "  F_12     = {:.2f}.\n"
           "  F_21     = {:.2f}.\n"
           "  F_22     = {:.2f}.\n"
           "  sigma_1  = {:.2f}.\n"
           "  sigma_2  = {:.2f}.\n"
           "  g_1      = {:.2f}.\n"
           "  g_2      = {:.2f}.\n"
           "  sigma_1o = {:.2f}.\n"
           "  sigma_2o = {:.2f}.\n"
           ).format(F_11, F_12, F_21, F_22, sigma_1, sigma_2, g_1, g_2,
                    sigma_1o, sigma_2o))

    sys.exit(3)
    
    # Set up matrices
    F = np.array([[F_11, F_12], [F_21, F_22]])
    Sigma = np.array([[sigma_1, 0], [0, sigma_2]])
    G = np.array([[g_1, 0], [0, g_2]])
    Sigma_o = np.array([[sigma_1o, 0], [0, sigma_2o]])
    

    ## Set up and plot the equilibrium PDF
    # Calculate the equilibrium PDF
    x = np.linspace(-1, 1, 500, endpoint=True) # u_1
    dx = x[1] - x[0]

    y = np.linspace(-1, 1, 500, endpoint=True) # u_2
    dy = y[1] - y[0]
    
    [true_cov_mtx, pdf] = p_eq(F, Sigma, x, y)
    pdf = pdf / np.sum((dx + dy) * pdf) # Normalize the PDF


    # Plot the equilibrium PDF
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(5, 4)

    ax = fig.add_subplot()
    cf = ax.contourf(x, y, pdf, levels=8, cmap='jet')

    ax.set_title('Equilibrium PDF')
    ax.set_xlabel(r'$u_1$')
    ax.set_ylabel(r'$u_2$')

    ax.set_aspect('equal', adjustable='box')

    cb = plt.colorbar(cf)

    # Save plot to file
    plot_file_name = '1b_eq_pdf.png'
    plt.savefig(plot_file_name, dpi=300)

    ## Simulate trajectories of 100 time units
    rng = default_rng(1)
    
    dt = 0.1
    start_t = 0.0
    end_t = 100.0
    t = np.arange(start_t, end_t + dt, dt)
    nsteps = np.size(t)
    

    # Initialize model state and covariance
    model_pst_mean = np.zeros([2, nsteps])
    model_pst_cov  = np.zeros([2, 2, nsteps])
    
    model_pst_mean[0, 0] = 0.5
    model_pst_mean[1, 0] = 0.5
    model_pst_cov[0, 0, 0] = 0.2
    model_pst_cov[1, 1, 0] = 0.5

    model_pri_mean = model_pst_mean
    model_pri_cov  = model_pst_cov
    

    # Initial condition
    trials[:, 0, 0] = 0.5
    trials[:, 1, 0] = 0.5

    # True state, observation of initial condition
    true_state = np.zeros([2, nsteps])
    true_state[0, 0] = 0.5
    true_state[1, 0] = 0.5

    obs = np.zeros([2, nsteps])
    stoch = rng.standard_normal(obs[:, 0])
    obs[:, 0] = G @ true_state + Sigma_o @ stoch

    for step in range(1, nsteps):

        # Calculate the prior stats and the true state
        [model_pri_mean[:, step], model_pri_cov[:, step]] \
            = filter_calc_pri(F, Sigma, dt,
                              model_pst_mean[:,step-1],
                              model_pst_cov[:,:,step-1])
        true_state[:, step] = advance_truth(rng, F, Sigma, dt,
                                            true_state[:, step-1])
        
        
    # Plot a single trial
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 3)

    ax = fig.add_subplot()
    ax.plot(t, trials[0, 0, :], color='#E69F00', label=r'$u_1$')
    ax.plot(t, trials[0, 1, :], color='#56B4E9', label=r'$u_2$')

    ax.set_title('Single Trajectory')
    ax.set_xlabel(r'$t$')
    ax.legend()

    # Save plot to file
    plot_file_name = '1b_traj.png'
    plt.savefig(plot_file_name, dpi=300)

    # Calculate ensemble statistics
    num_u1_mean = np.mean(trials[:, 0, :], axis=0)
    num_u2_mean = np.mean(trials[:, 1, :], axis=0)

    true_mean = 0.0

    num_u1_var = np.var(trials[:, 0, :], axis=0)
    num_u2_var = np.var(trials[:, 1, :], axis=0)
    num_u_cov  = np.zeros_like(t)
    for step in range(0, nsteps):
        num_u_cov[step] = np.cov(trials[:, :, step], rowvar=False)[0,1]

    true_u1_var = true_cov_mtx[0, 0]
    true_u2_var = true_cov_mtx[1, 1]
    true_u_cov  = true_cov_mtx[0, 1]

    # Plot ensemble statistics
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 6)

    gs = GridSpec(2, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, num_u1_mean, color='#E69F00', label=r'$\langle u_1 \rangle$')
    ax0.plot(t, num_u2_mean, color='#56B4E9', label=r'$\langle u_2 \rangle$')
    ax0.plot(t, true_mean*np.ones_like(t), color='#000000', linestyle='dashed')

    ax0.text(t[nsteps-1]/8, true_mean, r'$\langle u_1 \rangle_{eq}$',
             backgroundcolor='white')
    ax0.text(t[nsteps-1]/4, true_mean, r'$\langle u_2 \rangle_{eq}$',
             backgroundcolor='white')
    
    ax0.set_xlabel(r'$t$')
    ax0.set_title('Ensemble Means')
    ax0.legend()

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, num_u1_var, color='#E69F00', label=r'$Var(u_1)$')
    ax1.plot(t, num_u2_var, color='#56B4E9', label=r'$Var(u_2)$')
    ax1.plot(t, num_u_cov,  color='#009E73', label=r'$Cov(u_1,\ u_2)$')
    ax1.plot(t, true_u1_var*np.ones_like(t), color='#000000', linestyle='dashed')
    ax1.plot(t, true_u2_var*np.ones_like(t), color='#000000', linestyle='dashed')
    ax1.plot(t, true_u_cov*np.ones_like(t),  color='#000000', linestyle='dashed')

    ax1.text(t[nsteps-1]/8, true_u1_var, r'$Var(u_1)_{eq}$',
             backgroundcolor='white')
    ax1.text(t[nsteps-1]/4, true_u2_var, r'$Var(u_2)_{eq}$',
             backgroundcolor='white')
    ax1.text(t[nsteps-1]/2, true_u_cov,  r'$Cov(u_1,\ u_2)_{eq}$',
             backgroundcolor='white')

    ax1.set_xlabel(r'$t$')
    ax1.set_title('Ensemble Variances')
    ax1.legend()

    # Save plot to file
    plot_file_name = '1b_ens_stats.png'
    plt.savefig(plot_file_name, dpi=300)


def advance_truth(rng, F, Sigma, dt, state):

    stoch = rng.standard_normal(np.size(state))
    
    state_out = state + (F @ state) * dt \
        + np.sqrt(dt) * Sigma @ stoch
    
    return state_out

def filter_calc_pri(F, Sigma, dt, mean, cov):

    coeff_mtx = np.identity([2, 2]) + dt * F

    mean_out = coeff_mtx @ mean
    cov_out = coeff_mtx @ cov @ np.tranpose(coeff_mtx) + Sigma

    return [mean_out, cov_out]

if __name__ == "__main__":
    main(sys.argv[1:])

