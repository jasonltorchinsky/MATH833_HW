'''
Solves problem 3 of problem set 2.

Author: Jason Torchinsky
Date: October 5th, 2021
'''

import numpy as np
from numpy.random import default_rng

import sys, getopt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def main(argv):

    a     = 0.0
    omega = 0.0
    f     = 0.0
    sigma = 0.0

    helpStr = ('problem2.py -a <Decay parameter>'
               ' -w <Oscillation parameter>,'
               ' -f <Determinstic forcing>'
               ' -s <Stochastic strength>')
    try:
        opts, args = getopt.getopt(argv, 'a:w:f:s:', ['decay=',
                                                      'oscil=',
                                                      'force=',
                                                      'stoch='])
    except getopt.GetoptError:
        print(helpStr)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(helpStr)
            sys.exit()
        elif opt in ('-a', '--decay='):
            a = float(arg)
        elif opt in ('-w', '--oscil='):
            omega = float(arg)
        elif opt in ('-f', '--force='):
            f = float(arg)
        elif opt in ('-s', 'stoch='):
            sigma = float(arg)

    # Print startup message
    print(('Starting Problem 2 with a = {:8}, omega = {:8}, '
           'f = {:8}, sigma = {:8}.\n').format(a, omega, f, sigma))

    # Calculate mean, vaeriance, covariance of x
    t0   = 0.0
    tend = 10.0
    time = np.linspace(t0, tend, num=200, endpoint=True)

    mean_x0 = 1.0 + 1.0j
    var_x0  = 1.5
    pvar_x0 = 0.5

    mean_x  = np.exp((-a + 1.0j * omega) * time) * mean_x0 \
        + (f / (-a + 1.0j * omega)) * (np.exp((-a + 1.0j * omega) * time) - 1.0)
    var_x   = np.exp(-2.0 * a * time) * var_x0 \
        + (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * time))
    pvar_x  = np.exp((-a + 1.0j * omega) * time) * pvar_x0

    # Set up figure.
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(7.5, 6)
    
    gs = GridSpec(3, 1, figure=fig)
    
    # Plot of mean of x(t)
    ax0 = fig.add_subplot(gs[0,0])
    ax0.plot(time, np.real(mean_x), color = '#E69F00',
             label = 'Real Part')
    ax0.plot(time, np.imag(mean_x), color = '#56B4E9',
             label = 'Imaginary Part')
    
    ax0.set_ylabel('Mean x(t)')
    ax0.legend(loc='upper right')
    
    # Plot of variance of x(t)
    ax1 = fig.add_subplot(gs[1,0])
    ax1.plot(time, var_x, color = '#E69F00')
    
    ax1.set_ylabel('Variance x(t)')
    
    # Plot pseudo-variance of x(t)
    ax2 = fig.add_subplot(gs[2,0])
    ax2.plot(time, np.real(pvar_x), color = '#E69F00',
             label = 'Real Part')
    ax2.plot(time, np.imag(pvar_x), color = '#56B4E9',
             label = 'Imaginary Part')
    
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Pseudo-Variance x(t)')
    ax2.legend(loc='upper right')
    
    axs = [ax0, ax1, ax2]
    
    for ax in axs:
        ax.set_xscale('linear')
        ax.set_xlim(t0, tend)
        
        ax.set_yscale('linear')

       
            
    # Save plot to file
    plot_file_name = '3_stats.png'
    plt.savefig(plot_file_name,dpi=300)
        
if __name__ == "__main__":
    main(sys.argv[1:])
