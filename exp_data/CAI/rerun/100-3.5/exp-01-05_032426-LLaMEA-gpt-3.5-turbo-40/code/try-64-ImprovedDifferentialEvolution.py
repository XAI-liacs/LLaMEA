import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.diversity_threshold = 0.2  # Threshold for population diversity
        self.F_lower = 0.4  # Lower bound for F
        self.F_upper = 0.9  # Upper bound for F
        self.CR_lower = 0.6  # Lower bound for CR
        self.CR_upper = 1.0  # Upper bound for CR

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        diversity_history = []  # Track diversity over iterations
        for i in range(self.budget):
            indices = np.arange(self.budget)
            indices = indices[indices != i]
            a, b, c = np.random.choice(indices, 3, replace=False)

            # Adaptive control of F and CR based on population diversity
            current_diversity = np.mean(np.std(population, axis=0))
            if len(diversity_history) > 5:  # Allow adaptation after a few iterations
                if current_diversity < self.diversity_threshold:
                    self.F = min(self.F_upper, self.F * 1.1)
                    self.CR = max(self.CR_lower, self.CR * 0.9)
                else:
                    self.F = max(self.F_lower, self.F * 0.9)
                    self.CR = min(self.CR_upper, self.CR * 1.1)
            diversity_history.append(current_diversity)

            mutant = population[a] + self.F * (population[b] - population[c])
            crossover_mask = np.random.rand(self.dim) < self.CR
            offspring = np.where(crossover_mask, mutant, population[i])

            f_offspring = func(offspring)
            if f_offspring < func(population[i]):
                population[i] = offspring
            
            if f_offspring < self.f_opt:
                self.f_opt = f_offspring
                self.x_opt = offspring

        return self.f_opt, self.x_opt