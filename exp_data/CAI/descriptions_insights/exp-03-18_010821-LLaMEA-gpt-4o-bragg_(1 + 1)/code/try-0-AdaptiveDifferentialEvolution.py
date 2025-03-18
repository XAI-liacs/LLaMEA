import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, target_idx):
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        return np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)

    def crossover(self, target_vector, mutant_vector):
        crossover_vector = np.array([
            mutant_vector[i] if np.random.rand() < self.crossover_rate else target_vector[i]
            for i in range(self.dim)
        ])
        return crossover_vector

    def selection(self, target_idx, trial_vector, func):
        target_vector = self.population[target_idx]
        if func(trial_vector) < func(target_vector):
            return trial_vector
        else:
            return target_vector

    def __call__(self, func):
        self.initialize_population(func.bounds.lb, func.bounds.ub)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                mutant_vector = self.mutate(i)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                
                self.population[i] = self.selection(i, trial_vector, func)
                evaluations += 1
                
                # Dynamic parameter adjustment based on diversity
                if (evaluations % (self.population_size * 2)) == 0:
                    diversity = np.std(self.population)
                    if diversity < 0.1:
                        self.mutation_factor = min(1.0, self.mutation_factor + 0.1)
                    else:
                        self.mutation_factor = max(0.5, self.mutation_factor - 0.1)

        return min(self.population, key=func)