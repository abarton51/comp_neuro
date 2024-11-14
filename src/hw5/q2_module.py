import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from typing import List

"""
Modified version of the Hopfield network module for completing question 2
with non-random stored patterns.
"""

def preprocess_image(filename, n):
    patt2d = imread(filename, as_gray=True)
    patt2d = np.sign(patt2d / np.max(patt2d) - 0.5)
    return patt2d[:n, :n].reshape(1, n*n)

def plot_patterns(patterns, title):
    n = int(np.sqrt(patterns.shape[1]))
    num_patterns = patterns.shape[0]
    fig, axes = plt.subplots(1, num_patterns, figsize=(12, 3))
    for i, ax in enumerate(axes):
        ax.imshow(patterns[i].reshape(n, n), cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

def perturb_pattern(pattern, p):
    perturbed = pattern.copy()
    perturb_mask = np.random.rand(pattern.size) < p
    perturbed[perturb_mask] *= -1
    return perturbed

class HopfieldNetwork:
    def __init__(self, patterns: List[np.ndarray]):
        self.N = patterns.shape[1]
        self.M = np.zeros((self.N, self.N))
        self._compute_coupling_matrix(patterns)
    
    def _compute_coupling_matrix(self, patterns):
        for pattern in patterns:
            self.M += np.outer(pattern, pattern)
        self.M /= self.N
        np.fill_diagonal(self.M, 0)
    
    def run_network(self, initial_state, max_iterations=1000):
        current_state = initial_state.copy()
        states_over_time = [current_state.copy()]
        for iter in range(max_iterations):
            i = np.random.randint(0, self.N)
            current_state[i] = np.sign(np.dot(self.M[i], current_state))
            if current_state[i] == 0:
                current_state[i] = 1
            if (iter + 1) % (self.N) == 0 or (iter + 1) == 10 * self.N:
                states_over_time.append(current_state.copy())
        return states_over_time
