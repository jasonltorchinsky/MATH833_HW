'''
Solves problem 2 of problem set 2.

Author: Jason Torchinsky
Date: October 12th, 2021
'''

import numpy as np
from numpy.random import default_rng

import sys, getopt

from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt

def main(argv):

    n = 0

    helpStr = "problem1.py -n <Number of transitions>"
    try:
        opts, args = getopt.getopt(argv, "hn:", ["N="])
    except getopt.GetoptError:
        print(helpStr)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(helpStr)
            sys.exit()
        elif opt in ("-n", "--N"):
            n = int(arg)

    # Print startup message
    print(("Starting Problem 2 with n = {:8}.\n").format(n))
    
    rng = default_rng()

    # Set simulation parameters
    s_st = 2.0
    s_un = 1.0
    nu   = 0.75
    mu   = 0.25

    # Print system parameter values
    print(("Stable state value:   {:.4f}.\n"
           "Unstable state value: {:.4f}.\n"
           "Stable -> Unstable switching rate (nu): {:.4f}.\n"
           "Unstable -> Stable switching rate (mu): {:.4f}.\n").format(
               s_st, s_un, nu, mu))

    # Get the discrete-time Markov chain
    Y_n  = np.zeros([n])
    # 50-50 shot of starting in either state
    Y_n[0] = rng.uniform(low=0.0, high=1.0)
    if (Y_n[0] < 0.5):
        Y_n[0] = s_st
    else:
        Y_n[0] = s_un

    # Since this is a two-state Markov jump process, Y_n always just jumps
    # between s_un and s_st at every step
    for ii in range(1,n):
        Y_n[ii] = (s_st - Y_n[ii-1]) + s_un

    # Generate the transition times 
    tr_times = np.zeros([n])
    for ii in range(n):
        if Y_n[ii] == s_st:
            tr_times[ii] = rng.exponential(scale=1/nu)
        elif Y_n[ii] == s_un:
            tr_times[ii] = rng.exponential(scale=1/mu)

    t = np.cumsum(tr_times)

    # Find distributions for the switching times
    st_t = tr_times[Y_n == s_st] # Intervals where s = s_st
    un_t = tr_times[Y_n == s_un] # Intervals where s = s_un
    if n < 500:
        print("WARNING: Small n can lead to poor PDF approximations.\n")
    nbins = int(n/10)
    
    st_hist, st_bin_edges = np.histogram(st_t, bins=nbins, density=False)
    st_bin_centers = st_bin_edges[:-1] + (st_bin_edges[1] - st_bin_edges[0])/2.0
    st_pdf_num = np.cumsum(st_hist) / np.sum(st_hist)
    st_pdf_ana = 1.0 - np.exp(-nu * st_bin_centers)

    un_hist, un_bin_edges = np.histogram(un_t, bins=nbins, density=False)
    un_bin_centers = un_bin_edges[:-1] + (un_bin_edges[1] - un_bin_edges[0])/2.0
    un_pdf_num = np.cumsum(un_hist) / np.sum(un_hist)
    un_pdf_ana = 1.0 - np.exp(-mu * un_bin_centers)

    st_err = (1.0 / n) * np.sum(np.abs(st_pdf_num - st_pdf_ana))
    un_err = (1.0 / n) * np.sum(np.abs(un_pdf_num - un_pdf_ana))
    
    print(("One-norm error of numeric and analytic PDFs:\n"
           "P(T_st <= t): {:.8f}.\n"
           "P(T_un <= t): {:.8f}.\n").format(
               st_err, un_err ))

    # Find mean value of the trajectory
    num_mean_state = (1.0 / t[n-1]) * np.sum(tr_times * Y_n)
    ana_mean_state = (nu * s_un + mu * s_st) / (nu + mu)
    print(("(Numerical) Mean value of trajectory: {:.4f}.\n"
           "(Analytic)  Mean value of trajectory: {:.4f}.\n").format(
               num_mean_state, ana_mean_state))


    
    # Plot the first 20 transitions of the trajectory
    if n >= 20:
        n = 20
        
    fig = plt.figure()
    ax = fig.add_subplot()
    
    plt.step(t[0:n-1], Y_n[0:n-1], where='post')
    
    ax.set_yticks([s_st, s_un])
    ax.set_yticklabels([r'$s_{st}$', r'$s_{un}$'])

    # Save plot to file
    plot_file_name = '2b_traj.png'
    plt.savefig(plot_file_name,dpi=300)
    

if __name__ == "__main__":
    main(sys.argv[1:])
