import numpy as np

class HybridPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9  # Crossover probability
        self.f = 0.8   # Differential weight
        self.periodicity_weight = 0.1  # Weight for periodicity enforcement
        self.primary_population = np.random.uniform(0, 1, (self.population_size, self.dim))
        self.secondary_population = np.random.uniform(0, 1, (self.population_size, self.dim)) 
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        fitness = np.array([func(ind) for ind in self.primary_population])
        best_idx = np.argmin(fitness)
        best = self.primary_population[best_idx]
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.primary_population[np.random.choice(idxs, 3, replace=False)]
                mutant = self.mutate(a, b, c, lb, ub)
                trial = self.crossover(self.primary_population[i], mutant, best)
                
                # Periodicity enforcement
                trial = self.enforce_periodicity(trial, best, i)

                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    self.primary_population[i] = trial
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial

        return best

    def mutate(self, a, b, c, lb, ub):
        mutant = np.clip(a + self.f * (b - c), lb, ub)
        return mutant

    def crossover(self, target, mutant, best):
        cross_points = np.random.rand(self.dim) < (self.cr + 0.1 * np.random.rand())  # Adapted crossover
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def enforce_periodicity(self, solution, best, index):
        # Enhanced periodicity enforcement with dual population influence
        period = 2
        weight = (index % 10 + 1) / 10
        for start in range(0, self.dim, period):
            end = min(start + period, self.dim)
            avg = (solution[start:end] + best[start:end] + weight * self.secondary_population[index][start:end]) / 3
            solution[start:end] = np.mean(avg)
        return solution