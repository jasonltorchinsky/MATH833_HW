'''
Solves problem 7 of the midterm.

Author: Jason Torchinsky
Date: October 21st, 2021
'''

import numpy as np
from numpy.random import default_rng

import sys, getopt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy.stats as stats

def main(argv):

    ntrials = 1000
    

    helpStr = ("problem6c.py -n <Number of trials")
    try:
        opts, args = getopt.getopt(argv, "hn:", [])
    except getopt.GetoptError:
        print(helpStr)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(helpStr)
            sys.exit()
        elif opt == "-n":
            ntrials = int(arg)


    # Print startup message
    print(("Starting Problem 7 with:\n"
           "  ntrials =  {:8}.\n"
           ).format(ntrials))
    

    # Run the simulation a bunch of times
    sigma = 1.0
    rng = default_rng()
    nsteps = 10000
    dt = 0.01
    t = np.arange(0,nsteps+1)*dt

    xs = rng.uniform(low=0, high=2*np.pi, size=ntrials)
    ys = rng.uniform(low=0, high=2*np.pi, size=ntrials)

    #xs = rng.normal(loc=np.pi, scale=np.sqrt(np.pi), size=ntrials)
    #ys = rng.normal(loc=np.pi, scale=np.sqrt(np.pi), size=ntrials)
    
    x_moments = np.zeros([4, nsteps+1])
    y_moments = np.zeros([4, nsteps+1])

    x_moments[0,0] = np.mean(xs)
    x_moments[1,0] = stats.moment(xs, moment=2)
    x_moments[2,0] = stats.moment(xs, moment=3)
    x_moments[3,0] = stats.moment(xs, moment=4)


    y_moments[0,0] = np.mean(ys)
    y_moments[1,0] = stats.moment(ys, moment=2)
    y_moments[2,0] = stats.moment(ys, moment=3)
    y_moments[3,0] = stats.moment(ys, moment=4)

    print(('Initial stats:\n'
           '  x     :   {:.4f}   {:.4f}   {:.4f}   {:.4f}\n'
           '  y     :   {:.4f}   {:.4f}   {:.4f}   {:.4f}\n'
           ).format(x_moments[0,0], x_moments[1,0], x_moments[2,0], x_moments[3,0],
                    y_moments[0,0], y_moments[1,0], y_moments[2,0], y_moments[3,0]))

    for step in range(1,nsteps+1):

        time = t[step]
        
        [xs, ys] = advance_state(rng, sigma, time, dt, xs, ys)
        
        x_moments[0,step] = np.mean(xs)
        x_moments[1,step] = stats.moment(xs, moment=2)
        x_moments[2,step] = stats.moment(xs, moment=3)
        x_moments[3,step] = stats.moment(xs, moment=4)
        

        y_moments[0,step] = np.mean(ys)
        y_moments[1,step] = stats.moment(ys, moment=2)
        y_moments[2,step] = stats.moment(ys, moment=3)
        y_moments[3,step] = stats.moment(ys, moment=4)

    print(('Final stats:\n'
           '  x     :   {:.4f}   {:.4f}   {:.4f}   {:.4f}\n'
           '  y     :   {:.4f}   {:.4f}   {:.4f}   {:.4f}\n'
           ).format(x_moments[0,-1], x_moments[1,-1], x_moments[2,-1], x_moments[3,-1],
                    y_moments[0,-1], y_moments[1,-1], y_moments[2,-1], y_moments[3,-1]))

    # Guessed final stats
    exp_stats = np.zeros(4)
    exp_stats[0] = np.pi
    exp_stats[1] = np.pi
    exp_stats[2] = 0
    exp_stats[3] = 3*np.pi**2
        
    # Plot the evolution of the mean and variance
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 9)

    gs = GridSpec(4, 1, fig)

    # means
    ax0 = fig.add_subplot(gs[0,0])
    ax0.plot(t, x_moments[0,:], color='#E69F00', label='x')
    ax0.plot(t, y_moments[0,:], color='#56B4E9', label='y')

    ax0.plot(t, 0*t+exp_stats[0], color='#000000', linestyle='dashed')

    ax0.set_xlabel('t')
    ax0.set_xlim([t[0], t[-1]])

    ax0.set_ylabel('Mean')
    
    plt.legend()

    # variances
    ax1 = fig.add_subplot(gs[1,0])
    ax1.plot(t, x_moments[1,:], color='#E69F00', label='x')
    ax1.plot(t, y_moments[1,:], color='#56B4E9', label='y')
    
    ax1.plot(t, 0*t+exp_stats[1], color='#000000', linestyle='dashed')
    
    ax1.set_xlabel('t')
    ax1.set_xlim([t[0], t[-1]])

    ax1.set_ylabel('Variance')
    
    plt.legend()

    # skews
    ax2 = fig.add_subplot(gs[2,0])
    ax2.plot(t, x_moments[2,:], color='#E69F00', label='x')
    ax2.plot(t, y_moments[2,:], color='#56B4E9', label='y')
    
    ax2.plot(t, 0*t+exp_stats[2], color='#000000', linestyle='dashed')
    
    ax2.set_xlabel('t')
    ax2.set_xlim([t[0], t[-1]])

    ax2.set_ylabel('Skewness')
    
    plt.legend()

    # kurtosis
    ax3 = fig.add_subplot(gs[3,0])
    ax3.plot(t, x_moments[3,:], color='#E69F00', label='x')
    ax3.plot(t, y_moments[3,:], color='#56B4E9', label='y')
    
    ax3.plot(t, 0*t+exp_stats[3], color='#000000', linestyle='dashed')
    
    ax3.set_xlabel('t')
    ax3.set_xlim([t[0], t[-1]])

    ax3.set_ylabel('Kurtosis')
    
    plt.legend()
    
    # Save plot to file
    plot_file_name = '7_evo.png'
    plt.savefig(plot_file_name,dpi=300)
    

def advance_state(rng, sigma, t, dt, x, y):

    # Advance the state
    x_stoch = rng.standard_normal(np.size(x))
    y_stoch = rng.standard_normal(np.size(y))

    [u, v] = vel(t, x, y)
    
    x_out = x + u * dt + sigma * np.sqrt(dt) * x_stoch

    y_out = y + v * dt + sigma * np.sqrt(dt) * y_stoch


    # Enforce doubly-periodic boundary conditions
    x_out = np.remainder(x_out, 2*np.pi)
    y_out = np.remainder(y_out, 2*np.pi)
    
    return [x_out, y_out]

def vel(t, x, y):

    #u = -np.sin(x) * np.sin(y)
    #v = -np.cos(x) * np.cos(y)

    u = np.zeros_like(x)
    v = np.zeros_like(y)
    
    return [u, v]

if __name__ == "__main__":
    main(sys.argv[1:])
