'''
Solves problem 6(b) of the midterm.

Author: Jason Torchinsky
Date: October 20th, 2021
'''

import numpy as np
from numpy.random import default_rng

import sys, getopt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy.stats as stats

def main(argv):

    sigma = 1
    F = 0
    a = 0
    b = 0
    c = 0
    

    helpStr = ("problem6b.py -s <sigma parameter> -F <F parameter>"
               " -a <a parameter>, -b <b parameter>, -c <c parameter>")
    try:
        opts, args = getopt.getopt(argv, "hs:F:a:b:c:", [])
    except getopt.GetoptError:
        print(helpStr)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(helpStr)
            sys.exit()
        elif opt == "-s":
            sigma = float(arg)
        elif opt == "-F":
            F = float(arg)
        elif opt == "-a":
            a = float(arg)
        elif opt == "-b":
            b = float(arg)
        elif opt == "-c":
            c = float(arg)


    # Print startup message
    print(("Starting Problem 6b with:\n"
           "  sigma = {:.2f}.\n"
           "  F     = {:.2f}.\n"
           "  a     = {:.2f}.\n"
           "  b     = {:.2f}.\n"
           "  c     = {:.2f}.\n"
           ).format(sigma, F, a, b, c))

    # Calculate the equilibrium PDF
    x = np.linspace(-5,5,250,endpoint=True)
    dx = x[1] - x[0]
    pdf = p_eq(sigma, F, a, b, c, x)
    pdf = pdf / np.sum(dx * pdf) # Normalize the PDF


    # Plot the equilibrium PDF
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 6)

    gs = GridSpec(2, 1, figure=fig)

    ax0 = fig.add_subplot(gs[0,0])
    ax0.plot(x, pdf, color='#000000', label='True PDF')

    # Plot the Gaussian fit
    mean = np.sum(dx * x * pdf)
    var  = np.sum(dx * (x - mean)**2 * pdf)
    ax0.plot(x, stats.norm.pdf(x, mean, np.sqrt(var)),
            color='#000000',
            linestyle='dashed',
            label='Gaussian Fit')
    

    # Run the simulation a bunch of times
    rng = default_rng(1)
    ntrials = 10000
    nsteps = 1000
    dt = 0.01

    trials = np.zeros(ntrials)

    for step in range(1,nsteps+1):
        trials = advance_state(rng, sigma, F, a, b, c, dt, trials)

    # Plot a histogram for the trials
    ax0.hist(trials, bins=50, density=True,
             facecolor='#E69F00',
             label='Numerical PDF')

    plt.legend()

    ax0.set_ylabel('Equilibrium PDF')

    # Make semi-log plot
    ax1 = fig.add_subplot(gs[1,0])
    ax1.plot(x, pdf, color='#000000', label='True PDF')
    ax1.plot(x, stats.norm.pdf(x, mean, np.sqrt(var)),
            color='#000000',
            linestyle='dashed',
            label='Gaussian Fit')
    ax1.hist(trials, bins=50, density=True,
             facecolor='#E69F00',
             label='Numerical PDF')

    ax1.set_xlabel('x')
    ax1.set_ylabel('log(Equilibrium PDF)')
    ax1.set_ylim([0.001, 1.0])
    ax1.set_yscale('log')

    plt.legend()
    
    # Save plot to file
    plot_file_name = '6b_pdf.png'
    plt.savefig(plot_file_name,dpi=300)

    # Make a plot of a single trajectory
    t = np.arange(0,nsteps+1)*dt
    traj = np.zeros(nsteps+1)
    for step in range(1,nsteps+1):
        traj[step] = advance_state(rng, sigma, F, a, b, c, dt, traj[step-1])

    fig.clf()
    fig.set_size_inches(7.5, 4)

    ax = fig.add_subplot()
    ax.plot(t, traj, color='#000000')

    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_xlim([t[0], t[-1]])

    # Save plot to file
    plot_file_name = '6b_traj.png'
    plt.savefig(plot_file_name,dpi=300)
    

def p_eq(sigma, F, a, b, c, x):

    return np.exp((2/sigma) * (F*x + (a/2)*x**2 + (b/3)*x**3 - (c/4)*x**4))

def advance_state(rng, sigma, F, a, b, c, dt, x):

    stoch = rng.standard_normal(np.size(x))
    
    state_out = x + (F + a * x + b * x**2 - c * x**3) * dt \
        + sigma * np.sqrt(dt) * stoch
    
    return state_out

if __name__ == "__main__":
    main(sys.argv[1:])
