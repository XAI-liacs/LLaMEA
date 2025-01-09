import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        
    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                # Mutation
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, -5.0, 5.0)
                
                # Crossover
                trial = np.copy(self.population[i])
                crossover_points = np.random.rand(self.dim) < self.crossover_rate
                trial[crossover_points] = mutant[crossover_points]

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial
            
            # Adapt mutation factor and crossover rate
            self.mutation_factor = 0.5 + 0.3 * np.random.rand()
            self.crossover_rate = 0.5 + 0.4 * np.random.rand()

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]