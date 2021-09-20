'''
Solves problem 3 of problem set 1.

Author: Jason Torchinsky
Date: September 20th, 2021
'''

import numpy as np
from scipy.integrate import simps

import sys, getopt

def main(argv):

    tol = 0

    helpStr = "problem1.py -tol <Error tolerance>"
    try:
        opts, args = getopt.getopt(argv, "ht:", ["tol="])
    except getopt.GetoptError:
        print(helpStr)
        sys.exit(2)

    for opt, arg in opts:
        if opt == "-h":
            print(helpStr)
            sys.exit()
        elif opt in ("-t", "--tol"):
            tol = float(arg)

    # Print startup message
    print(("Starting Problem 3 with tol = {:8}.\n").format(tol))
            
    

    # Print true value of integral
    x_min, x_max, nx = (0, 1, 100)
    y_min, y_max, ny = (0, 2, 100)
    xx = np.linspace(x_min, x_max, nx)
    yy = np.linspace(y_min, y_max, ny)
    zz = np.zeros([nx, ny])
    for x_idx in range(nx):
        x = xx[x_idx]
        for y_idx in range(ny):
            y = yy[y_idx]
            zz[x_idx, y_idx] = f(x, y)
        
    truInt = simps([simps(zz_x, xx) for zz_x in zz], yy) 
    print(("'True' approximation of integral: {:6.4}.\n").format(truInt))

    # We repeat the experiment a whole bunch
    ntrials = 1000
    results = np.zeros(ntrials)
    N = int(np.ceil(tol**(-1)*0.824955))
    for trial in range(ntrials):
        # Obtain samples, numerical value of integral
        Xs = np.random.default_rng().uniform(0, 1, N)
        Ys = np.random.default_rng().uniform(0, 2, N)
        Zs = np.zeros([N, N])
        for x_idx in range(N):
            x = Xs[x_idx]
            for y_idx in range(N):
                y = Ys[y_idx]
                Zs[x_idx, y_idx] = f(x, y)
                results[trial] = 2.0 * np.mean(Zs)

    numInt = np.mean(results)

    print(("Numerical approxmation of integral: {:6.4}.\n").format(numInt))

def f(x, y):
    return np.exp(x)*np.power(y, x)

if __name__ == "__main__":
    main(sys.argv[1:])
