import numpy as np

class RefinedQuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def apply_gate(self, individual):
        return individual * np.random.choice([-1, 1], size=self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = [func(ind) for ind in self.population]
            best_idx = np.argmin(fitness)
            best_individual = self.population[best_idx]
            elite_idx = np.argsort(fitness)[:int(0.1*self.budget)]  # Selecting top 10% individuals
            for idx in range(self.budget):
                if idx not in elite_idx:
                    self.population[idx] = 0.5 * (self.population[idx] + self.apply_gate(best_individual))
        return best_individual