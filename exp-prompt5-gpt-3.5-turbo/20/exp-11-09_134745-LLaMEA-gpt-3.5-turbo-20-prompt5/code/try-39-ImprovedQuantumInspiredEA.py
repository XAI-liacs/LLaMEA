import numpy as np

class ImprovedQuantumInspiredEA:
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
            for idx in range(self.budget):
                rnd_idx = np.random.choice([i for i in range(self.budget) if i != idx])
                if func(self.population[rnd_idx]) < func(self.population[idx]):
                    self.population[idx] = self.population[rnd_idx] + np.random.uniform(-0.1, 0.1, self.dim)
                else:
                    self.population[idx] = 0.5 * (self.population[idx] + self.apply_gate(best_individual))
        return best_individual