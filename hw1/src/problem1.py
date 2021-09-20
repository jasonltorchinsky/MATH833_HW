'''
Solves problem 1 of problem set 1.

Author: Jason Torchinsky
Date: September 20th, 2021
'''

import numpy as np
import sys, getopt

def main(argv):

    N = 0

    helpStr = "problem1.py -N <Number of Points>"
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
            N = int(arg)

    # Print startup message
    print(("Starting Problem 1 with N = {:8}.\n").format(N))
            
    # Mean, covariance of joint distribution
    mu = np.array([0, 0])
    R  = np.array([[1, 0.5], [0.5, 1]])

    # Print analytic mean and variances
    truMu = mu[0] + R[0,1]/R[1,1]*(1.0 - mu[1])
    truR  = R[0,0] - R[0,1]/R[1,1]*R[1,0]
    print(("Conditional distribution analytic mean is {:6.4}.\n"
           "Conditional distribution analytic variance is {:6.4}.\n")
          .format(truMu, truR))

    # Repeat this Monte Carlo calculation a whole lot
    ntrials = 1000
    results = np.zeros([ntrials, 3]) # mean, variance, number of valid samples
    for trial in range(ntrials):
        # Get samples from joint, conditional distribution
        samps = np.random.default_rng().multivariate_normal(mu, R, N)
        
        condMask = (samps[:, 1] > 0.9) & (samps[:, 1] < 1.1)
        condSamps = samps[condMask, :]

        # Get stats from conditional distribution
        condMu = np.mean(condSamps[:, 0])
        condR  = np.var(condSamps[:,0])
        condN = np.size(condSamps[:,0])

        results[trial,:] = [condMu, condR, condN]


    condMu = np.mean(results[:,0])
    condR  = np.var(results[:,1])
    condN  = np.mean(results[:,2])
    
    print(("Conditional distribution numerical mean is {:6.4}.\n"
           "Conditional distribution numerical variance is {:6.4}.\n")
          .format(condMu, condR))

    # Calculate confidence interval
    conf = 1.96 * np.sqrt(condR / condN)
    confInt  = np.array([condMu - conf, condMu + conf])
    
    print(("Conditional distribution numerical confidence interval is "
           "[{:6.4}, {:6.4}].\n").format(confInt[0], confInt[1]))

if __name__ == "__main__":
    main(sys.argv[1:])
