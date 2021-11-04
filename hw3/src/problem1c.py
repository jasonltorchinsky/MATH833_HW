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
    
    # Set up matrices
    F = np.array([[F_11, F_12], [F_21, F_22]])
    Sigma = np.array([[sigma_1, 0], [0, sigma_2]])
    G = np.array([[g_1, 0], [0, g_2]])
    Sigma_o = np.array([[sigma_1o, 0], [0, sigma_2o]])
    

    ## Simulate trajectory of 100 time units
    rng_model = default_rng(1)
    rng_truth = default_rng(2)
    rng_obs = default_rng(3)
    

    # Time-stepping for model
    start_t = 0.0
    end_t = 100.0

    dt_model = 0.25
    t_model = np.arange(start_t, end_t + dt_model, dt_model)
    nsteps_model = np.size(t_model)

    dt_obs = 0.25
    t_obs = np.arange(start_t, end_t + dt_obs, dt_obs)
    nsteps_obs = np.size(t_obs)

    # To have different time-step sizes for obsevations and the models,
    # We need to keep track of values for two time-series

    # Prior stats and one copy of the truth signal
    model_pri_mean = np.zeros([2,nsteps_model])
    model_pri_cov  = np.zeros([2,2,nsteps_model])
    
    model_pri_mean[:,0]  = np.array([0.5,0.5])
    model_pri_cov[:,:,0] = np.array([[0.2,0.0],[0.0,0.5]])

    truth_pri = np.zeros([2,nsteps_model])
    truth_pri[:,0] = np.array([0.5,0.5])

    
    # Posterior stats, one copy of the truth signal, and the observations
    model_pst_mean = np.zeros([2,nsteps_obs])
    model_pst_cov  = np.zeros([2,2,nsteps_obs])
    
    model_pst_mean[:,0]  = model_pri_mean[:,0]
    model_pst_cov[:,:,0] = model_pri_cov[:,:,0]

    truth_pst = np.zeros([2,nsteps_obs])
    truth_pst[:,0] = truth_pri[:,0]
    

    obs = np.zeros([2,nsteps_obs])
    stoch = rng_obs.standard_normal(np.shape(obs[:,0]))
    obs[:, 0] = G @ truth_pst[:,0] + Sigma_o @ stoch

    step_model = 1     # Step number for model
    step_obs = 1       # Step number for observation
    obs_flag = False   # Whether we had an observation last step
    for time in t_model[1:]:

        # If we just made an observation, update the prior stats from the
        # previous posterior        
        if (obs_flag):
            [model_pri_mean[:,step_model], model_pri_cov[:,:,step_model]] \
                = filter_calc_pri(F, Sigma, dt_model,
                                  model_pst_mean[:,step_obs-1],
                                  model_pst_cov[:,:,step_obs-1])
        else: # Otherwise update prior stats from prior stats
            [model_pri_mean[:,step_model], model_pri_cov[:,:,step_model]] \
                = filter_calc_pri(F, Sigma, dt_model,
                                  model_pri_mean[:,step_model-1],
                                  model_pri_cov[:,:,step_model-1])

        # Update true state
        truth_pri[:,step_model] = advance_truth(rng_truth, F, Sigma, dt_model,
                                              truth_pri[:,step_model-1])
            
        # Time for an observation!
        if np.abs(np.mod(time, dt_obs)) < dt_model/10: # Time for an observation!
            # Update truth for observation time grid
            truth_pst[:,step_obs] = truth_pri[:,step_model]

            # Generate observation
            stoch = rng_obs.standard_normal(np.shape(obs[:,step_obs]))
            obs[:,step_obs] = G @ truth_pst[:,step_obs] + Sigma_o @ stoch

            # Assimilate observation
            gain_mtx = model_pri_cov[:,:,step_model] @ np.transpose(G) \
                @ np.linalg.inv(G @ model_pri_cov[:,:,step_model] @ np.transpose(G) + Sigma_o)

            model_pst_mean[:,step_obs] = model_pri_mean[:, step_model] \
                + gain_mtx @ (obs[:, step_obs] - G @ model_pri_mean[:, step_model])
            model_pst_cov[:,:,step_obs] = (np.identity(2) - gain_mtx @ G) @ model_pri_cov[:,:,step_model]
            
            # Advance step number, set observation flag
            step_model = step_model + 1
            step_obs = step_obs + 1
            obs_flag = True

        else: # Not time for an observation!
            # Advance step number, set observation flag
            step_model = step_model + 1
            obs_flag = False

    
    # Plot ensemble statistics
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 9)

    gs = GridSpec(5, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t_model, model_pri_mean[0,:],
             color='#000000', linestyle='solid',
             label=r'$\langle u_1 \rangle$ Prior')
    ax0.plot(t_obs, model_pst_mean[0,:],
             color='#E69F00', linestyle='dashed',
             label=r'$\langle u_1 \rangle$ Posterior')
    
    
    ax0.set_xlabel(r'$t$')
    ax0.legend()

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t_model, model_pri_mean[1,:],
             color='#000000', linestyle='solid',
             label=r'$\langle u_2 \rangle$ Prior')
    ax1.plot(t_obs, model_pst_mean[1,:],
             color='#56B4E9', linestyle='dashed',
             label=r'$\langle u_2 \rangle$ Posterior')
    
    
    ax1.set_xlabel(r'$t$')
    ax1.legend()

    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(t_model, model_pri_cov[0,0,:],
             color='#000000', linestyle='solid',
             label=r'$Var(u_1)$ Prior')
    ax2.plot(t_obs, model_pst_cov[0,0,:],
             color='#E69F00', linestyle='dashed',
             label=r'$Var(u_1)$ Posterior')
    
    ax2.set_xlabel(r'$t$')
    ax2.legend()

    ax3 = fig.add_subplot(gs[3, 0])
    ax3.plot(t_model, model_pri_cov[1,1,:],
             color='#000000', linestyle='solid',
             label=r'$Var(u_2)$ Prior')
    ax3.plot(t_obs, model_pst_cov[1,1,:],
             color='#56B4E9', linestyle='dashed',
             label=r'$Var(u_2)$ Posterior')
    
    ax3.set_xlabel(r'$t$')
    ax3.legend()

    ax4 = fig.add_subplot(gs[4, 0])
    ax4.plot(t_model, model_pri_cov[0,1,:],
             color='#000000', linestyle='solid',
             label=r'$Cov(u_1,\ u_2)$ Prior')
    ax4.plot(t_obs, model_pst_cov[0,1,:],
             color='#009E73', linestyle='dashed',
             label=r'$Cov(u_1,\ u_2)$ Posterior')
    
    ax4.set_xlabel(r'$t$')
    ax4.legend()

    # Save plot to file
    plot_file_name = '1e_stats.png'
    plt.savefig(plot_file_name, dpi=300)


    # Print out root-mean-square error
    RMSE_u1_pri = np.sqrt(1/nsteps_model
                          * np.sum((truth_pri[0,:] - model_pri_mean[0,:])**2))
    RMSE_u2_pri = np.sqrt(1/nsteps_model
                          * np.sum((truth_pri[1,:] - model_pri_mean[1,:])**2))
    RMSE_u1_pst = np.sqrt(1/nsteps_obs
                          * np.sum((truth_pst[0,:] - model_pst_mean[0,:])**2))
    RMSE_u2_pst = np.sqrt(1/nsteps_obs
                          * np.sum((truth_pst[1,:] - model_pst_mean[1,:])**2))
    
    print(('Root-mean-squared error:\n'
           '  u_1 prior:     {:.4f}.\n'
           '  u_1 posterior: {:.4f}.\n'
           '  u_2 prior:     {:.4f}.\n'
           '  u_2 posterior: {:.4f}.\n'
           ).format(RMSE_u1_pri, RMSE_u1_pst, RMSE_u2_pri, RMSE_u2_pst))

    # Calculate  correlations
    autocorr_u1_pri    = np.correlate(model_pri_mean[0,:], model_pri_mean[0,:],
                                      mode='full')
    autocorr_u1_pri    = autocorr_u1_pri[int(autocorr_u1_pri.size/2):]
    
    autocorr_u2_pri    = np.correlate(model_pri_mean[1,:], model_pri_mean[1,:],
                                      mode='full')
    autocorr_u2_pri    = autocorr_u2_pri[int(autocorr_u2_pri.size/2):]
    
    corr_u1_u2_pri     = np.correlate(model_pri_mean[0,:], model_pri_mean[1,:],
                                      mode='full')
    corr_u1_u2_pri     = corr_u1_u2_pri[int(corr_u1_u2_pri.size/2):]
    
    corr_u1_pri_truth  = np.correlate(model_pri_mean[0,:], truth_pri[0,:],
                                      mode='full')
    corr_u1_pri_truth  = corr_u1_pri_truth[int(corr_u1_pri_truth.size/2):]
    
    corr_u2_pri_truth  = np.correlate(model_pri_mean[1,:], truth_pri[1,:],
                                      mode='full')
    corr_u2_pri_truth  = corr_u2_pri_truth[int(corr_u2_pri_truth.size/2):]
    
    autocorr_u1_pst    = np.correlate(model_pst_mean[0,:], model_pst_mean[0,:],
                                      mode='full')
    autocorr_u1_pst    = autocorr_u1_pst[int(autocorr_u1_pst.size/2):]
    
    autocorr_u2_pst    = np.correlate(model_pst_mean[1,:], model_pst_mean[1,:],
                                      mode='full')
    
    corr_u1_u2_pst     = np.correlate(model_pst_mean[0,:], model_pst_mean[1,:],
                                      mode='full')
    
    corr_u1_pst_truth  = np.correlate(model_pst_mean[0,:], truth_pst[0,:],
                                      mode='full')
    corr_u1_pst_truth  = corr_u1_pst_truth[int(corr_u1_pst_truth.size/2):]
    
    corr_u2_pst_truth  = np.correlate(model_pst_mean[1,:], truth_pst[1,:],
                                      mode='full')
    corr_u2_pst_truth  = corr_u2_pst_truth[int(corr_u2_pst_truth.size/2):]

    # Plot correlations
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 4)

    gs = GridSpec(2, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t_model, corr_u1_pri_truth,
             color='#E69F00', linestyle='solid',
             label=r'$Corr(\langle u_1 \rangle$ Prior, $u_1$ Truth$)$')
    ax0.plot(t_model, corr_u2_pri_truth,
             color='#56B4E9', linestyle='solid',
             label=r'$Corr(\langle u_2 \rangle$ Prior, $u_2$ Truth$)$')
    
    
    ax0.set_xlabel(r'$t$')
    ax0.set_ylabel('Cross-Correlation')
    ax0.set_ylim([-10,10])
    ax0.legend()

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t_model, corr_u1_pst_truth,
             color='#E69F00', linestyle='solid',
             label=r'$Corr(\langle u_1 \rangle$ Posterior, $u_1$ Truth$)$')
    ax1.plot(t_model, corr_u2_pst_truth,
             color='#56B4E9', linestyle='solid',
             label=r'$Corr(\langle u_2 \rangle$ Posterior, $u_2$ Truth$)$')
    
    
    ax1.set_xlabel(r'$t$')
    ax1.set_ylabel('Cross-Correlation')
    ax1.set_ylim([-10,10])
    ax1.legend()
    

    # Save plot to file
    plot_file_name = '1e_corr.png'
    plt.savefig(plot_file_name, dpi=300)
    
def advance_truth(rng, F, Sigma, dt, state):

    stoch = rng.standard_normal(np.size(state))
    
    state_out = state + (F @ state) * dt \
        + np.sqrt(dt) * Sigma @ stoch
    
    return state_out

def filter_calc_pri(F, Sigma, dt, mean, cov):

    coeff_mtx = np.identity(2) + dt * F

    mean_out = coeff_mtx @ mean
    cov_out = coeff_mtx @ cov @ np.transpose(coeff_mtx) + Sigma

    return [mean_out, cov_out]

if __name__ == "__main__":
    main(sys.argv[1:])

