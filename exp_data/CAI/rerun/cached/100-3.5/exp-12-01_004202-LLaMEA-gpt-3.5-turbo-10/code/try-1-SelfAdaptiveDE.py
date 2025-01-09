import numpy as np

class SelfAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim, pop_size=10, f=0.5, cr=0.9, mutation_prob=0.1):
        super().__init__(budget, dim, pop_size, f, cr)
        self.mutation_prob = mutation_prob

    def mutate(self, population, target_idx):
        parents = self.select_parents(population, target_idx)
        best = population[np.argmin([func(ind) for ind in population])]
        mutant = population[target_idx] + self.f * (best - population[target_idx]) + self.mutation_prob * np.random.uniform(-5.0, 5.0, self.dim)
        return np.clip(mutant, -5.0, 5.0)