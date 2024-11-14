import numpy as np
from typing import Dict
import matplotlib.pyplot as plt

class Hopfield:
    model_name = 'Hopfield Network'

    def __init__(self, param: Dict):
        default_keys = ['N', 'k', 'iterations', 'seed']
        if not all(key in param for key in default_keys):
            print("Warning: Some default keys are missing, using default values.")
        self.N = param.get('N', 16)
        self.k = param.get('k', 3)
        self.iterations = param.get('iterations', 1000)

        self.seed = param.get('seed', 42)
        np.random.seed(self.seed)

        self.patterns = self._generate_patterns()

        self.M = np.zeros((self.N, self.N))
        self._compute_coupling_matrix()

    def _generate_patterns(self):
        return np.random.choice([-1, 1], size=(self.k, self.N))

    def _compute_coupling_matrix(self):
        for mu in range(self.k):
            self.M += np.outer(self.patterns[mu], self.patterns[mu])
        self.M /= self.N
        np.fill_diagonal(self.M, 0)

    def compute_outer_product(self, x):
        """Non numpy implementation (deprecated and replace with np.outer)"""
        print("Warning: Consider using np.outer instead of this method. They do the same thing but np.outer is faster.")
        outer_product = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                outer_product[i, j] = x[i] * x[j]
        return outer_product

    def check_convergence(self, x):
        """Check if the current state has converged to a stored pattern."""
        return np.all(np.sign(np.dot(self.M, x)) == x)

    def compute_average_difference(self, original, final):
        """Compute the average difference between the current state and a stored pattern."""
        return np.mean(np.abs(original - final))

    def perturb_pattern(self, x, p=0.1):
        """Perturb a stored pattern by flipping bits/signs with probability p."""
        perturbed_pattern = x.copy()
        perturbed_pattern[np.random.rand(self.N) < p] *= -1
        return perturbed_pattern

    def run_network(self, initial_state, max_iterations=1000):
        """Async update for the Hopfield network."""
        current_state = initial_state.copy()
        for _ in range(max_iterations):
            i = np.random.randint(0, self.N)
            current_state[i] = np.sign(np.dot(self.M[i], current_state))
            if current_state[i] == 0:
                current_state[i] = 1
        return current_state

    def simulate(self, p=None):
        if p is None:
            p = np.linspace(0, 1, 20)
        v_diff_avgs = []
        for perturb_prob in p:
            pattern_diff = []
            for mu in range(self.k):
                original_pattern = self.patterns[mu]
                perturbed_pattern = self.perturb_pattern(original_pattern, perturb_prob)
                final_state = self.run_network(perturbed_pattern)
                diff = self.compute_average_difference(original_pattern, final_state)
                pattern_diff.append(diff)

            v_diff_avg = np.mean(pattern_diff)
            v_diff_avgs.append(v_diff_avg)

        return p, v_diff_avgs

class HopfieldSimulator:
    def __init__(self, num_networks: 100, param: Dict):
        self.num_networks = num_networks
        self.N_values = param.get('N', [16])
        self.k_values = param.get('k', [3])
        self.iterations = param.get('iterations', 100)
        self.p = param.get('p', np.linspace(0, 1, 20))

    def run_simulation(self, show_plot=True):
        results = {}
        for N in self.N_values:
            for k in self.k_values:
                param = {'N': N, 'k': k, 'iterations': 10 * N}
                v_diff_list = []
                for _ in range(self.num_networks):
                    network = Hopfield(param)
                    p_values, v_diff = network.simulate(self.p)
                    v_diff_list.append(v_diff)

                avg_v_diff = np.mean(v_diff_list, axis=0)
                results[(N, k)] = (p_values, avg_v_diff)

        if show_plot:
            self.plot_results(results)
        return results

    def plot_results(self, results):
        plt.figure(figsize=(12, 8))

        for (N, k), (p_values, avg_v_diff) in results.items():
            plt.plot(p_values, avg_v_diff, label=f'N={N}, k={k}')

        plt.xlabel('Perturbation Probability (p)')
        plt.ylabel('Average Difference ($\\bar{v}_{diff}$)')
        plt.title('Pattern Retrieval Accuracy as a Function of Perturbation Probability')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()
