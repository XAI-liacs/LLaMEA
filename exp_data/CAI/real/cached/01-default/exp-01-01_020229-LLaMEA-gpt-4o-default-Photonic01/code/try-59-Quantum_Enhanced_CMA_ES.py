import numpy as np

class Quantum_Enhanced_CMA_ES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(3 * np.log(dim))
        self.sigma = 0.5  # Initial step size
        self.mu = self.population_size // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2) ** 2 + self.mueff))
        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.csigma = (self.mueff + 2) / (dim + self.mueff + 5)
        self.dsigma = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.csigma
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)
        
        mean = np.random.uniform(lb, ub, self.dim)
        covariance_matrix = np.eye(self.dim)
        path_c = np.zeros(self.dim)
        path_sigma = np.zeros(self.dim)
        eigen_decomposition = np.eye(self.dim)
        eigen_values = np.ones(self.dim)
        evaluations = 0
        
        while evaluations < self.budget:
            samples = np.random.multivariate_normal(np.zeros(self.dim), covariance_matrix, self.population_size)
            samples *= self.sigma
            samples = mean + samples
            
            # Quantum-inspired sampling adjustment
            samples += np.random.uniform(-0.1, 0.1, samples.shape)
            samples = np.clip(samples, lb, ub)
            
            fitness = np.array([func(sample) for sample in samples])
            evaluations += self.population_size
            
            indices = np.argsort(fitness)
            selected = samples[indices[:self.mu]]
            selected_weighted = np.dot(self.weights, selected)
            
            mean = selected_weighted
            path_sigma = (1 - self.csigma) * path_sigma + np.sqrt(self.csigma * (2 - self.csigma) * self.mueff) / self.sigma * np.dot(eigen_decomposition, mean - selected_weighted)
            
            E_norm = np.sqrt(2) * self.dim / 4
            hsig = np.linalg.norm(path_sigma) / np.sqrt(1 - (1 - self.csigma) ** (2 * evaluations / self.population_size)) / E_norm < 1.4 + 2 / (self.dim + 1)
            path_c = (1 - self.cc) * path_c + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) / self.sigma * (selected_weighted - mean)
            
            cov_update = np.dot(selected.T, np.dot(np.diag(self.weights), selected)) - np.outer(mean, mean)
            covariance_matrix = (1 - self.c1 - self.cmu) * covariance_matrix + self.c1 * (np.outer(path_c, path_c) + (1 - hsig) * self.cc * (2 - self.cc) * covariance_matrix) + self.cmu * cov_update
            
            self.sigma *= np.exp((self.csigma / self.dsigma) * (np.linalg.norm(path_sigma) / E_norm - 1))
            
            if evaluations >= self.budget:
                break
            
            if evaluations % (10 * self.population_size) == 0:
                eigen_values, eigen_decomposition = np.linalg.eigh(covariance_matrix)
        
        best_index = np.argmin(fitness)
        return samples[best_index], fitness[best_index]