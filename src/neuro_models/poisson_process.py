import numpy as np
from typing import Dict
from scipy.stats import expon
import matplotlib.pyplot as plt

class PoissonProcess:
    def __init__(self, param: Dict):
        if param["firing rate"] <= 0:
            raise ValueError("rate must be positive real number")
        self.firing_rate = param["firing rate"] # Firing rate of neuron
        self.duration = param["T"]
        self.realizations = param["realizations"]
        self.refractoriness = param.get("refractoriness", None)
        self.isi = []
        self.spike_times = []
        self.num_spikes = []
    
    def init_expon_rv(self):
        self.expon_rv = expon(scale=1./self.firing_rate)
    
    def simulate_interspike_intervals(self, num_intervals=200) -> float:
        self.isi = []
        self.spike_times = []
        self.num_spikes = []
        for i in range(self.realizations):
            tau = expon.rvs(scale=1/self.firing_rate, size=num_intervals)
            self.isi.append(tau)
            spikes = np.cumsum(tau)
            filtered_spikes = spikes[spikes < self.duration]
            self.spike_times.append(filtered_spikes)
            self.num_spikes.append(len(filtered_spikes))
            
    def simulate_refractory_isi(self, num_intervals=1000) -> None:
        self.isi = []
        self.spike_times = []
        self.num_spikes = []
        for i in range(self.realizations):
            tau = []
            current_time = 0
            while len(tau) < num_intervals:
                interval = expon.rvs(scale=1/self.firing_rate)
                while interval <= self.refractoriness:
                    interval = expon.rvs(scale=1/self.firing_rate)
                tau.append(interval)
                current_time += interval
                if current_time >= self.duration:
                    break

            self.isi.append(tau)
            spikes = np.cumsum(tau)
            filtered_spikes = spikes[spikes < self.duration]
            self.spike_times.append(filtered_spikes)
            self.num_spikes.append(len(filtered_spikes))
    
    def compute_fano_factor(self):
        self.fano_factor = np.var(self.num_spikes) / np.mean(self.num_spikes)
        return self.fano_factor
    
    def plot_interspike_intervals(self, scale="log") -> None:
        all_isi = np.concatenate(self.isi)
        plt.figure(figsize=(10, 5))
        plt.hist(all_isi, bins=100, density=True, alpha=0.7, color='b', label='Histogram of ISI')
        x = np.linspace(0, np.max(all_isi), 10000)
        plt.plot(x, expon.pdf(x, scale=1/self.firing_rate), 'r-', lw=2, label='Exponential PDF')
        ylabel = "Density"
        if scale=="log":
            plt.yscale('log')
            ylabel += " (log scale)"
        plt.title('Interspike Interval Histogram')
        plt.xlabel('Interspike Interval (s)')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()
