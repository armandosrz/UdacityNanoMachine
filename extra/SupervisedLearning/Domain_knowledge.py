'''
K Nearest Neighbor implementations for the points
    data = [[(1,6), 7], [(2,4), 8], [(3,7), 16], [(6,8),44], [(7,1),50], [(8,4),68]]
Seeking the point
    q = (4,2)
Using two different distance formulas: Manhathan and euclidean_distance
If the the subsequent elements to k have the same distance as k, those are
also included.


'''


import math
import numpy as np


def manh_dist(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean_dist(a,b):
    return math.sqrt(math.pow(a[0]-b[0], 2)+ math.pow(a[1]-b[1], 2))

def KNN(k, data, q, dist):
    r = sorted([(dist(a[0],q), index) for index, a in enumerate(data)])
    avg = [data[b][1] for a, b in r[:k]]

    # Check if the distance value is the same for the elements after k
    # if so add them to the list to get the averages.
    max_k_value = r[k-1][0]
    start_index = k
    while True:
        if r[start_index][0] == max_k_value:
            avg.append(data[r[start_index][1]][1])
            start_index += 1
        else: break
    average = np.mean(avg)
    print 'Average for {} Neighbors, using {} as distance is: {}'.format(
        k, dist.__name__, average
    )


if __name__ == '__main__':
    data = [[(1,6), 7], [(2,4), 8], [(3,7), 16], [(6,8),44], [(7,1),50], [(8,4),68]]
    q = (4,2)
    KNN(1, data, q, manh_dist)
    KNN(3, data, q, manh_dist)
    KNN(1, data, q, euclidean_dist)
    KNN(3, data, q, euclidean_dist)
