import numpy as np

class HybridPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9  # Crossover probability
        self.f = 0.8   # Differential weight
        self.periodicity_weight = 0.1  # Weight for periodicity enforcement
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = self.adaptive_mutate(a, b, c, lb, ub, best)
                trial = self.crossover(population[i], mutant, best)
                
                # Periodicity enforcement
                trial = self.enforce_periodicity(trial, best)

                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial

        return best

    def adaptive_mutate(self, a, b, c, lb, ub, best):
        # Introduce adaptive mutation strategy
        adapt_factor = np.random.rand() * (1 - np.linalg.norm(best - (a + b + c) / 3) / np.linalg.norm(ub - lb))
        mutant = np.clip(a + self.f * (b - c) + adapt_factor, lb, ub)
        return mutant

    def crossover(self, target, mutant, best):
        cross_points = np.random.rand(self.dim) < (self.cr + 0.1 * np.random.rand())  # Adapted crossover
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def enforce_periodicity(self, solution, best):
        # Enhanced periodicity enforcement by using the best solution as a guide
        period = 2
        for start in range(0, self.dim, period):
            end = min(start + period, self.dim)
            avg = (solution[start:end] + best[start:end]) / 2
            solution[start:end] = np.mean(avg)
        return solution