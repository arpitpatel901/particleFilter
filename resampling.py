
from numpy.random import random
import numpy as np

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill (1.0 / len(weights))

def multinomal_resample(weights):
    """
    Compute the cumulative sum of the normalized weights. 
    This gives you an array of increasing values from 0 to 1
    
    Idea : Due to cum sum large number occupy more size [0,1] , more
    chance of rnadom number falling in it
    run time : O(n:pretty shit)
    """
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off errors
    return np.searchsorted(cumulative_sum, random(len(weights)))

def residual_resample(weights):
    """
    Idea : Multiply weight(eg:0.0012) with number of particles(eg:3000)
           scaled weight 3.6, use 3 samples of that particle.
           Select the rest(need total N selections) , by using simpler 
           scheme like multinomial on fractionaly part of number(3.6-3 =0.6)
    run_time = O(n)
    """
    N = len(weights)
    indexes = np.zeros(N, 'i')

    # take int(N*w) copies of each weight
    w = np.asarray(weights)
    num_copies = (N*w).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]): # make n copies
            indexes[k] = i
            k += 1

    # use multinormial resample on the residual to fill up the rest.
    residual = w - num_copies
    # get fractional part
    residual /= sum(residual)
    # normalize
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1. # ensures sum is exactly one
    indexes[k:N] = np.searchsorted(cumulative_sum, random(N-k))

    return indexes

def stratified_resample(weights):
    """
    dividing the cumulative sum into N equal sections, 
    and then selects one particle randomly from each section .

    guarantees that each sample is between 0 and 2/N apart
    """
    N = len(weights)

    # make N subdivisions, chose a random position within each one
    positions = (random(N) + range(N)) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def systematic_resample(weights):
    """
    As with stratified resampling the space is divided
    into N divisions. We then choose a random offset to use for all of the divisions, 
    ensuring that each sample is exactly 1/N apart.
    """
    N = len(weights)
    # make N subdivisions, choose positions
    # with a consistent random offset
    positions = (np.arange(N) + random()) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    
    i, j = 0, 0
    
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes