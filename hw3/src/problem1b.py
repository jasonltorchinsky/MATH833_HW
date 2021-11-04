'''
Solves problem 1b of problem set 3.

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
    

    helpStr = ("problem1.py --F_11 <F_11 parameter> --F_12 <F_12 parameter> "
               "--F_21 <F_21 parameter> --F_22 <F_22 parameter> "
               "--sigma_1 <sigma_1 parameter>  --sigma_2 <sigma_2 parameter>")
    try:
        # Dummy short names
        opts, args = getopt.getopt(argv, "hq:w:e:r:t:y:", ['F_11=',
                                                           'F_12=',
                                                           'F_21=',
                                                           'F_22=',
                                                           'sigma_1=',
                                                           'sigma_2='])
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

    # Print startup message
    print(("Starting Problem 1 with:\n"
           "  F_11    = {:.2f}.\n"
           "  F_12    = {:.2f}.\n"
           "  F_21    = {:.2f}.\n"
           "  F_22    = {:.2f}.\n"
           "  sigma_1 = {:.2f}.\n"
           "  sigma_2 = {:.2f}.\n"
           ).format(F_11, F_12, F_21, F_22, sigma_1, sigma_2))

    # Set up matrices
    F = np.array([[F_11, F_12], [F_21, F_22]])
    Sigma = np.array([[sigma_1, 0], [0, sigma_2]])
    

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
    plot_file_name = '1e_eq_pdf.png'
    plt.savefig(plot_file_name, dpi=300)

    ## Simulate trajectories of 100 time units
    rng = default_rng(1)
    
    dt = 0.1
    start_t = 0.0
    end_t = 100.0
    t = np.arange(start_t, end_t + dt, dt)
    nsteps = np.size(t)
    
    ntrials = 500

    trials = np.zeros([ntrials, 2, nsteps]) # Trial, state variable (u_1, u_2),
                                            # step

    # Initial condition
    trials[:, 0, 0] = 0.5
    trials[:, 1, 0] = 0.5

    for step in range(1, nsteps):
        for trial in range(ntrials) :
            trials[trial, :, step] = advance_state(rng, F, Sigma, dt,
                                                   trials[trial, :, step-1])

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
    plot_file_name = '1e_traj.png'
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
    plot_file_name = '1e_ens_stats.png'
    plt.savefig(plot_file_name, dpi=300)

def p_eq(F, Sigma, x, y):

    
    # Covariance of equil distribution, and its inverse
    det_F = np.linalg.det(F)
    tr_F  = np.trace(F)
    cov = np.zeros([2, 2])
    cov[0,0] = - (Sigma[0,0]**2 * (det_F + F[1,1]**2)
                  + Sigma[1,1]**2 * F[0,1]**2) / (2 * tr_F * det_F)
    cov[0,1] = (Sigma[0,0]**2 * F[1,0] * F[1,1]
                + Sigma[1,1]**2 * F[0,0] * F[0,1]) / (2 * tr_F * det_F)
    cov[1,0] = cov[0,1]
    cov[1,1] = - (Sigma[0,0]**2 * F[1,0]**2
                  + Sigma[1,1]**2 * (det_F + F[0,0]**2)) / (2 * tr_F * det_F)
    
    cov_inv = np.linalg.inv(cov)

    # Normalization coefficient for PDF
    norm = 1.0 / np.sqrt(2.0 * np.pi * np.linalg.det(cov))
    
    nx = np.size(x)
    ny = np.size(y)

    pdf_eq = np.zeros([nx, ny])

    for x_idx in range(nx):
        x_crd = x[x_idx]
        for y_idx in range(ny):
            y_crd = y[y_idx]
            vec = np.array([[x_crd], [y_crd]])

            # Mean is zero
            pdf_eq[x_idx, y_idx] = norm \
                * np.exp(-0.5 * np.transpose(vec) @ cov_inv @ vec)

    return [cov, pdf_eq]

def advance_state(rng, F, Sigma, dt, state):

    stoch = rng.standard_normal(np.size(state))
    
    state_out = state + (F @ state) * dt \
        + np.sqrt(dt) * Sigma @ stoch
    
    return state_out

if __name__ == "__main__":
    main(sys.argv[1:])

