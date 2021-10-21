'''
Solves problem 6(c) of the midterm.

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
    

    helpStr = ("problem6c.py -s <sigma parameter> -F <F parameter>"
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
    print(("Starting Problem 6c with:\n"
           "  sigma = {:.2f}.\n"
           "  F     = {:.2f}.\n"
           "  a     = {:.2f}.\n"
           "  b     = {:.2f}.\n"
           "  c     = {:.2f}.\n"
           ).format(sigma, F, a, b, c))



    # Calculate the mean and variance of the equilibrium PDF
    x = np.linspace(-5,5,250,endpoint=True)
    dx = x[1] - x[0]
    pdf = p_eq(sigma, F, a, b, c, x)
    pdf = pdf / np.sum(dx * pdf) # Normalize the PDF
    mean = np.sum(dx * x * pdf)
    var  = np.sum(dx * (x - mean)**2 * pdf)
    

    # Run the simulation a bunch of times
    rng = default_rng(1)
    ntrials = 10000
    nsteps = 10000
    dt = 0.001
    t = np.arange(0,nsteps+1)*dt

    trials = np.sqrt(9) * rng.standard_normal(ntrials) + 10
    means = np.zeros(nsteps+1)
    variances = np.zeros(nsteps+1)

    means[0] = np.mean(trials)
    variances[0] = np.var(trials)

    for step in range(1,nsteps+1):
        trials = advance_state(rng, sigma, F, a, b, c, dt, trials)
        means[step] = np.mean(trials)
        variances[step] = np.var(trials)
        
    # Plot the evolution of the mean and variance
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 4)

    ax = fig.add_subplot()
    ax.plot(t, means, color='#E69F00', label='Mean')
    ax.plot(t, 0*t+mean, color='#E69F00', linestyle='dashed')
    
    ax.plot(t, variances, color='#56B4E9', label='Variance')
    ax.plot(t, 0*t+var, color='#56B4E9', linestyle='dashed')

    ax.set_xlabel('t')
    ax.set_xlim([t[0], t[-1]])
    
    plt.legend()

    # Save plot to file
    plot_file_name = '6c_evo.png'
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
