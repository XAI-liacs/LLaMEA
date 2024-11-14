import numpy as np

class EnhancedDynamicCMAStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.sigma = 1.0
        self.mean = np.random.uniform(-5.0, 5.0, dim)
        self.C = np.identity(dim)
        self.p_sigma = np.zeros(dim)
        self.p_c = np.zeros(dim)
        self.history = []

    def __call__(self, func):
        for _ in range(self.budget):
            z = np.random.normal(0, 1, self.dim)
            x = self.mean + self.sigma * np.dot(self.C, z)
            if func(x) < func(self.mean):
                self.mean = x
            c_diff = (1 - 1/(4*self.dim)) * self.p_c + np.sqrt(1/(2*self.dim)) * z
            self.C = np.dot(np.dot(np.identity(self.dim) - np.outer(c_diff, c_diff) * 2/(np.linalg.norm(c_diff)**2), self.C), np.identity(self.dim))
            p_sigma_new = (1 - 1/(2*self.dim)) * self.p_sigma + np.sum(self.C * z) / np.linalg.norm(self.C * z)
            self.p_sigma = p_sigma_new
            self.sigma *= np.exp((np.linalg.norm(self.p_sigma) - 1) / np.sqrt(self.dim + 3) * 0.3)
            if func(x) < func(self.mean):
                self.sigma *= 1.1
            else:
                self.sigma *= 0.9
            self.mean += np.random.normal(0, 0.1, self.dim)
            self.history.append(func(self.mean))

        if len(self.history) > self.dim * 2:
            if np.mean(self.history[-self.dim:])/np.mean(self.history[-2*self.dim:-self.dim]) > 1:
                self.sigma *= 1.2
            else:
                self.sigma *= 0.8
            self.history = self.history[-2*self.dim:]

        return self.mean