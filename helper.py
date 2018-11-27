from numpy.random import uniform
import numpy as np
import scipy

def create_uniform_particles(x_range, y_range, t_range, N):
    """
    Create uniform distribution of particles over a region

    Args:
        x_range             : [x_min ,x_max]
        y_range             : [y_min ,y_max]
        t_range(heading)    : [t_min ,t_max]
        N : Number of particles

    Examples:
        >>> ret = create_uniform_particles((0,1), (0,1), (0, np.pi*2), 4)
    """
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(t_range[0], t_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles

def create_gaussian_particles(mean, std, N):
    """
    Create gaussian distribution of particles over a region

    Args:
        x_range             : [x_min ,x_max]
        y_range             : [y_min ,y_max]
        t_range(heading)    : [t_min ,t_max]
        N : Number of particles

    Examples:
        >>> ret = create_gaussian_particles((0,1), (0,1), (0, np.pi*2), 4)
    """
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (np.random.randn(N) * std[0])
    particles[:, 1] = mean[1] + (np.random.randn(N) * std[1])
    particles[:, 2] = mean[2] + (np.random.randn(N) * std[2])
    particles[:, 2] %= 2 * np


def predict(particles, u, std, dt=1.):
    """ 
    Update the belief in the system state : 
    Move particles according to control input u ( omega, velocity)
    with noise Q (std(omega), std(velocity) 

    Args: 
        particles           : Particles you want to update state for
        u                   : Control input [omega , velocity]
        std                 : Assumed error in particle movement [std_dev(omega),std_dev(v)]
    """

    N = len(particles)
    # update heading
    particles[:, 2] += u[0] + (np.random.randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi
    # move in the (noisy) commanded direction
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist

def update(particles, weights, z, R, landmarks):
    """
    Sequential Importance Sampling

    Args :
        particles           : Particles to be sampled
        weights             : Probability that each particle represents the true position of the robot
        z                   : Z value for calculation of pdf for normal distribution [0,z]
        R                   : Residual of the distance measurement
        landmarks           : Known location from the sensor measurement

    """
    # Initilize the particle weights
    weights.fill(1.)

    for i, landmark in enumerate(landmarks):
        # measure the eucledian distance betn particles and actual positions
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1) 
        #importance density : weight the particles according to how well they match the measurements
        weights *= scipy.stats.norm(distance, R).pdf(z[i]) # make a normal continuous random variable and take its pdf

    weights += 1.e-300 # Approximation to avoid round-off to zero
    weights /= sum(weights) # normalize the weights


def estimate(particles, weights):
    """

    Returns state estimate based on particle positions
    Assumption : Unimodal(tracking only one state)

    Args :
        particles           : Particles to be sampled
        weights             : Probability that each particle represents the true position of the robot

    """
    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def neff(weights):
    """
    Effective N . Only resample if the value is above threshold
    Measures number of particles that meaningfully contribute to pdf

    Avoids resampling when sensor hasnt recieved any new information that 
    can be of use .
    """
    return 1. / np.sum(np.square(weights))

def simple_resample(particles, weights):
    """
    Multinomial resampling algorithm
    Samples from current particle set N times, making new set of 
    particles from the set. P(selecting a particle) propotional to its weight
    """
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    # finds indices inside sorted array cumulative_sum, such that if N was inserted , it would 
    # preserve the order(ascending) of the array 
    indexes = np.searchsorted(cumulative_sum, np.random.random(N)) 
    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)



