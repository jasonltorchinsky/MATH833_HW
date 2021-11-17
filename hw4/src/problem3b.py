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

    
    #Perform the Kalman filter
    rng = default_rng(1)
    
    dt = 0.01
    start_t = 0.0
    end_t = 100.0
    t = np.arange(start_t, end_t + dt, dt)
    nsteps = np.size(t)

    # Initialize model
    model = dict()
    model['a'] = a
    model['sigma'] = sigma
    model['f_pri'] = np.zeros([nsteps])
    model['f_pst'] = np.zeros([nsteps])
    model['u_pri'] = np.zeros([nsteps])
    model['u_pst'] = np.zeros([nsteps])
    model['Sigma_pri'] = np.zeros([2, 2, nsteps])
    model['Sigma_pst'] = np.zeros([2, 2, nsteps])

    model['Sigma_pst'][:,:,0] = np.array([[1, 1],[1, 50]])

    # Initialize truth
    truth = dict()
    truth['a'] = a
    truth['f'] = f
    truth['sigma'] = sigma
    truth['u'] = np.zeros_like(t)

    # Initialize observations
    obs = dict()
    obs['u'] = np.zeros([nsteps])
    G = np.array([[g, 0]])
    Sigma_o = np.array([sigma_o])
    
    
    for step in range(1, nsteps):
        # Advance model, truth, and observe truth
        truth['u'][step] = \
            advance_truth(rng, truth['a'],
                          truth['f'],
                          truth['sigma'],
                          truth['u'][step-1],
                          dt)
        [model['u_pri'][step],
         model['f_pri'][step],
         model['Sigma_pri'][:,:,step]] = \
             advance_model(model['a'],
                           model['f_pst'][step-1],
                           model['sigma'],
                           model['u_pst'][step-1],
                           model['Sigma_pst'][:,:,step-1],
                           dt)
        
        obs['u'][step] = \
            observe_truth(rng, G, Sigma_o, truth['u'][step], truth['f'])
        
        # Assimilate observation into model
        [model['u_pst'][step],
         model['f_pst'][step],
         model['Sigma_pst'][:,:,step]] = \
             assimilate_obs(model['u_pri'][step], model['f_pri'][step],
                            model['Sigma_pri'][:,:,step],
                            obs['u'][step], G, Sigma_o)
        
    
    # Plot the truth versus posterior mean
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 6)

    gs = GridSpec(2, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, model['u_pst'], color='#E69F00', label='Model')
    ax0.plot(t, truth['u'], color='#56B4E9', label='Truth')
    
    ax0.set_ylabel(r'$u$')
    
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, model['f_pst'], color='#E69F00', label='Model')
    ax1.plot(t, truth['f']+np.zeros_like(t), color='#56B4E9', label='Truth')
    
    ax1.set_ylabel(r'$f$')

    axs = [ax0, ax1]
    for ax in axs:
        ax.set_xlabel(r'$t$')
        ax.legend()
    
    fig.suptitle('Posterior Means')

    
    # Save plot to file
    plot_file_name = '3c_traj.png'
    plt.savefig(plot_file_name, dpi=300)


    # Plot the posterior covariances
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 3)

    gs = GridSpec(1, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, model['Sigma_pst'][0,0,:], color='#E69F00', label=r'$Var(u)$')
    ax0.plot(t, model['Sigma_pst'][0,1,:], color='#56B4E9', label=r'$Cov(u,f)$')
    ax0.plot(t, model['Sigma_pst'][1,1,:], color='#009E73', label=r'$Var(f)$')

    axs = [ax0]
    for ax in axs:
        ax.set_xlabel(r'$t$')
        ax.legend()
    
    fig.suptitle('Posterior Covariances')

    
    # Save plot to file
    plot_file_name = '3c_cov.png'
    plt.savefig(plot_file_name, dpi=300)

    sys.exit(2)
    
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

def advance_truth(rng, a, f, sigma, u, dt):

    stoch = rng.standard_normal([1])
    
    u_out = u \
        + (-a * u + f) * dt \
        + sigma * np.sqrt(dt) * stoch[0]

    f_out = f
    
    return u_out

def advance_model(a, f, sigma, u, Sigma, dt):
    
    A = np.array([[-a, 1],[0, 0]])
    I = np.identity(2)
    state = np.array([[u],[f]])
    sigma_mtx = np.array([[sigma, 0],[0, 0]])

    state_out = (I + dt * A) @ state
    Sigma_out = (I + dt * A) @ Sigma @ np.transpose(I + dt * A) + sigma_mtx
    
    
    return [state_out[0], state_out[1], Sigma_out]

def observe_truth(rng, G, Sigma_o, true_u, true_f):

    stoch = rng.standard_normal([1])

    true = np.array([[true_u],[true_f]])
    
    return G @ true + Sigma_o @ stoch

def assimilate_obs(u_pri, f_pri, Sigma_pri, obs_u, G, Sigma_o):

    gain = Sigma_pri @ np.transpose(G) \
        @ np.linalg.inv(G @ Sigma_pri @ np.transpose(G) + Sigma_o)

    state_pri = np.array([[u_pri], [f_pri]])
    state_pst = state_pri + gain @ (obs_u - G @ state_pri)
    Sigma_pst = (np.identity(2) - gain @ G) @ Sigma_pri

    return [state_pst[0], state_pst[1], Sigma_pst]


if __name__ == "__main__":
    main(sys.argv[1:])

