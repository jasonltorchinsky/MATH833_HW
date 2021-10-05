'''
Solves problem 1 of problem set 2.

Author: Jason Torchinsky
Date: October 4th, 2021
'''

import numpy as np
from numpy.random import default_rng

import sys, getopt

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def main(argv):

    n = 0
    do_plot = False

    helpStr = "problem1.py --N <Number of Points> --do-plot"
    try:
        opts, args = getopt.getopt(argv, "hn:", ["N=","do-plot"])
    except getopt.GetoptError:
        print(helpStr)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(helpStr)
            sys.exit()
        elif opt in ("-n", "--N"):
            n = int(arg)
        elif opt == "--do-plot":
            do_plot = True

    # Print startup message
    print(("Starting Problem 1 with N = {:8}.\n").format(n))
            
    # Set initial condition and transition matrix
    rng = default_rng()
    P = np.asarray([[0.0    , 1.0/3.0, 2.0/3.0],
                    [1.0/2.0, 1.0/2.0, 0.0    ],
                    [1.0/4.0, 1.0/2.0, 1.0/4.0]])
    state = (rng.integers(low = 1, high = 4, size = 1))[0]
    
    traj = np.zeros(n+1)
    traj[0] = state
    
    # Progress the Markov Chain
    for step in range(1,n+1):
        tr_sel = rng.uniform(0.0, 1.0) # Select transition
        state_idx = state - 1 # Index of the state
        if (tr_sel < P[state_idx,0]):
            state = 1
        elif (P[state_idx,0] < tr_sel
              and tr_sel < np.sum(P[state_idx,[0,1]])):
            state = 2
        else:
            state = 3
            
        traj[step] = state

    # Get equilibrium distribution by taking last half of steps
    counts = np.bincount(traj[int(n/2):].astype(int))
    print( 'Numerically estimated distribution: '
           + str(counts[1:]/np.sum(counts)) )

    if do_plot:
        # Set up figure.
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(7.5, 6)

        gs = GridSpec(2, 3, figure=fig)

        ax0 = fig.add_subplot(gs[0,:])
        ax0.plot(range(n+1), traj, color = '#E69F00')
        ax0.set_xlim(0,n)

        ax1 = fig.add_subplot(gs[1,0])
        ax1.plot(range(0,101), traj[0:101], color = '#E69F00')
        ax1.set_xlim(0,100)

        ax2 = fig.add_subplot(gs[1,1])
        ax2.plot(range(5000,5101), traj[5000:5101], color = '#E69F00')
        ax2.set_xlim(5000,5101)

        ax3 = fig.add_subplot(gs[1,2])
        ax3.plot(range(9900,n+1), traj[9900:n+1], color = '#E69F00')
        ax3.set_xlim(9900,n+1)

        axs = [ax0, ax1, ax2, ax3]
        
        for ax in axs:
            ax.set_xscale('linear')
            
            ax.set_yscale('linear')
            ax.set_yticks([1, 2, 3])
            pad = 0.1
            ax.set_ylim(1.0-pad, 3.0+pad)

        for ax in axs[0:2]:
            ax.set_ylabel('State')

        for ax in axs[1:]:
            ax.set_xlabel('Step Number')

       
            
        # Save plot to file
        plot_file_name = '1a_traj.png'
        plt.savefig(plot_file_name,dpi=300)
        
if __name__ == "__main__":
    main(sys.argv[1:])
