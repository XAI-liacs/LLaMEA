import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.F_min, self.F_max = 0.5, 0.9
        self.CR_min, self.CR_max = 0.1, 0.9

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        evaluations = self.pop_size
        best_idx = np.argmin(fitness)
        best = population[best_idx].copy()

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Adaptive parameter setting with chaotic dynamics
                F = self.F_min + (self.F_max - self.F_min) * np.abs(np.sin(evaluations))
                CR = self.CR_min + (self.CR_max - self.CR_min) * np.abs(np.cos(evaluations))

                parents = np.random.choice(self.pop_size, 3, replace=False)
                x1, x2, x3 = population[parents]
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
                
                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < CR
                trial[crossover_mask] = mutant[crossover_mask]
                
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial.copy()
                
                if evaluations >= self.budget:
                    break

        return best, fitness[best_idx]