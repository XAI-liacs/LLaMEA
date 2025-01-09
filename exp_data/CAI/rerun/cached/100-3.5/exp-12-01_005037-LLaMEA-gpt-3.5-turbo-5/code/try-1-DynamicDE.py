import numpy as np

class DynamicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.NP = 10
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        best_sol = np.random.uniform(-5.0, 5.0, self.dim)
        best_fit = func(best_sol)
        for _ in range(self.budget):
            trial_population = []
            for _ in range(self.NP):
                a, b, c = np.random.choice(range(self.NP), 3, replace=False)
                mutant_vec = best_sol + self.F * (best_sol - trial_population[a]) + self.F * (trial_population[b] - trial_population[c])
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vec = np.where(crossover_mask, mutant_vec, best_sol)
                trial_fit = func(trial_vec)
                if trial_fit < best_fit:
                    best_sol = trial_vec
                    best_fit = trial_fit
                trial_population.append(trial_vec)
        return best_sol