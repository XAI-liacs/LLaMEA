import numpy as np

class HybridPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9  # Crossover probability
        self.f = 0.5 + np.random.rand() * 0.3  # Adaptive Differential weight
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
                mutant = self.mutate(a, b, c, lb, ub)
                trial = self.crossover(population[i], mutant)
                
                # Enhanced Periodicity enforcement
                trial = self.enforce_periodicity(trial, period=4)

                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial

        return best

    def mutate(self, a, b, c, lb, ub):
        mutant = np.clip(a + self.f * (b - c), lb, ub)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.cr
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)
    
    def enforce_periodicity(self, solution, period=2):
        for start in range(0, self.dim, period):
            end = min(start + period, self.dim)
            solution[start:end] = np.mean(solution[start:end])
        return solution