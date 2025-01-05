import numpy as np

class QuantumTunnelingAdaptiveCuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_nests = max(10, min(50, budget // 10))
        self.nests = None
        self.best_nest = None
        self.best_fitness = float('inf')
        self.pa = 0.25  # discovery rate of alien eggs/solutions
        self.sigma = 0.1
        self.gamma = 0.9
        self.alpha = 0.01

    def levy_flight(self):
        beta = 1.5
        sigma_u = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v) ** (1 / beta)
        return step

    def evaluate_nests(self, func):
        fitness = np.array([func(n) for n in self.nests])
        for i, f in enumerate(fitness):
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_nest = self.nests[i]
        return fitness

    def quantum_tunneling(self):
        for i in range(self.num_nests):
            if np.random.rand() < self.alpha:
                step_size = np.random.randn(self.dim)
                self.nests[i] += step_size * self.sigma * (self.best_nest - self.nests[i])
                self.nests[i] = np.clip(self.nests[i], self.lb, self.ub)

    def adaptive_cuckoo_search(self, func):
        for _ in range(self.budget // self.num_nests):
            for i in range(self.num_nests):
                step = self.levy_flight()
                new_cuckoo = self.nests[i] + self.gamma * step * (self.nests[i] - self.best_nest)
                new_cuckoo = np.clip(new_cuckoo, self.lb, self.ub)
                new_fitness = func(new_cuckoo)
                if new_fitness < self.evaluate_nests(func)[i]:
                    self.nests[i] = new_cuckoo
            abandon_idxs = np.random.rand(self.num_nests) < self.pa
            self.nests[abandon_idxs] = self.lb + (self.ub - self.lb) * np.random.rand(np.sum(abandon_idxs), self.dim)
            self.quantum_tunneling()

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.nests = self.lb + (self.ub - self.lb) * np.random.rand(self.num_nests, self.dim)
        self.evaluate_nests(func)

        evaluations = 0
        while evaluations < self.budget:
            self.adaptive_cuckoo_search(func)
            evaluations += self.num_nests

        return self.best_nest, self.best_fitness