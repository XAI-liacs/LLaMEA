import numpy as np

class QILFS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 2)
        self.alpha = 0.1  # Exploration factor in Levy flight
        self.beta = 1.5   # Parameter for Levy distribution
        self.population = None
        self.population_values = None
        self.best_solution = None
        self.best_value = np.inf

    def levy_flight(self, scale=0.1):
        sigma1 = ((np.math.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)) /
                  (np.math.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        u = np.random.normal(0, sigma1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v) ** (1 / self.beta)
        return step * scale
    
    def quantum_update(self, current, best):
        return current + np.random.normal(0, 1, self.dim) * (best - current)

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.population_values = np.full(self.population_size, np.inf)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                candidate = self.population[i]
                
                if np.random.rand() < 0.5:
                    candidate = self.quantum_update(candidate, self.best_solution if self.best_solution is not None else candidate)
                else:
                    candidate += self.levy_flight(self.alpha * (ub - lb))
                
                candidate = np.clip(candidate, lb, ub)
                candidate_value = func(candidate)
                evaluations += 1

                if candidate_value < self.population_values[i]:
                    self.population_values[i] = candidate_value
                    self.population[i] = candidate.copy()

                if candidate_value < self.best_value:
                    self.best_value = candidate_value
                    self.best_solution = candidate.copy()

        return self.best_solution, self.best_value