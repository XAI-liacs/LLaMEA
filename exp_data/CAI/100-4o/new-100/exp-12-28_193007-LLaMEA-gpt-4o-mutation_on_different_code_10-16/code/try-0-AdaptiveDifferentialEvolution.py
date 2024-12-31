import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt, idx = np.min(fitness), np.argmin(fitness)
        self.x_opt = population[idx].copy()
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                evaluations += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                    
                # Adapt mutation factor and crossover rate
                if evaluations % (self.population_size * 5) == 0:
                    if self.f_opt < np.min(fitness):
                        self.mutation_factor = min(1.0, self.mutation_factor + 0.05)
                        self.crossover_rate = max(0.1, self.crossover_rate - 0.05)
                    else:
                        self.mutation_factor = max(0.1, self.mutation_factor - 0.05)
                        self.crossover_rate = min(0.9, self.crossover_rate + 0.05)
                
                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt