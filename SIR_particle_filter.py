
from numpy.linalg import norm
from numpy.random import randn
from numpy.random import random
import scipy.stats
from numpy.random import seed
import numpy as np
import matplotlib.pyplot as plt

from resampling import systematic_resample , resample_from_index 
from helper import create_gaussian_particles , create_uniform_particles, predict , update,neff ,estimate

def run_pf1(N, iters=18, sensor_std_err=.1,do_plot=True, plot_particles=False,xlim=(0, 20), ylim=(0, 20),initial_x=None):

    landmarks = np.array([[-1, 2], [5, 10], [12,14], [18,21]])
    NL = len(landmarks)

    plt.figure()

    # create particles and weights
    if initial_x is not None:
        particles = create_gaussian_particles(mean=initial_x, std=(5, 5, np.pi/4), N=N)
    else:
        particles = create_uniform_particles((0,20), (0,20), (0, 6.28), N)
    
    weights = np.zeros(N)
    
    if plot_particles:
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000)/np.sqrt(N)
        plt.scatter(particles[:, 0], particles[:, 1],alpha=alpha, color='g')
    
    xs = []
    robot_pos = np.array([0., 0.])

    for x in range(iters):
        robot_pos += (1, 1)

        # distance from robot to each landmark
        zs = (norm(landmarks - robot_pos, axis=1) +
        (randn(NL) * sensor_std_err))

        # move diagonally forward to (x+1, x+1)
        predict(particles, u=(0.00, 1.414), std=(.2, .05)) # assumed to be moving in a straight line (1-X,1-Y) with no angle change
        
        # incorporate measurements
        # weight of the particle is computed as the probability that it matches
        # the Gaussian of the sensor error model. Further the particle from measured distance,less likely it is a good 
        # representation
        update(particles, weights, z=zs, R=sensor_std_err,landmarks=landmarks)
        
        # resample if too few effective particles
        if neff(weights) < N/2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
        mu, var = estimate(particles, weights)
        xs.append(mu)

        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 1],color='k', marker=',', s=1)
        
        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+',color='k', s=180, lw=3)
        
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')

    xs = np.array(xs)
    #plt.plot(xs[:, 0], xs[:, 1])
    plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
    plt.show()


if __name__ == "__main__":
    seed(2)
    run_pf1(N=5000, plot_particles=True, initial_x=(1,1, np.pi/4))  