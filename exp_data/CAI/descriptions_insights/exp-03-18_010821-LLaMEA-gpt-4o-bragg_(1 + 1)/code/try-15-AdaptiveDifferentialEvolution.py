import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_factor = 10
        self.population_size = self.base_population_factor * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = None

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, target_idx, lb, ub):
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        if np.random.rand() > 0.5:
            mutant_vector = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        else:
            mutant_vector = self.population[a] + self.mutation_factor * (self.population[b] - self.population[target_idx])
        return np.clip(mutant_vector, lb, ub)

    def crossover(self, target_vector, mutant_vector):
        crossover_vector = np.array([
            mutant_vector[i] if np.random.rand() < self.crossover_rate else target_vector[i]
            for i in range(self.dim)
        ])
        return crossover_vector

    def selection(self, target_idx, trial_vector, func):
        target_vector = self.population[target_idx]
        trial_score = func(trial_vector)
        target_score = func(target_vector)
        # Weighted selection based on performance
        alpha = 0.5
        return trial_vector if trial_score < target_score else (alpha * trial_vector + (1 - alpha) * target_vector)

    def __call__(self, func):
        self.initialize_population(func.bounds.lb, func.bounds.ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                mutant_vector = self.mutate(i, func.bounds.lb, func.bounds.ub)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                
                self.population[i] = self.selection(i, trial_vector, func)
                evaluations += 1
                
                if (evaluations % (self.population_size * 2)) == 0:
                    diversity = np.std(self.population)
                    if diversity < 0.1:
                        self.mutation_factor = min(1.0, self.mutation_factor + 0.15)
                        self.crossover_rate = min(1.0, self.crossover_rate + 0.05)
                        self.population_size = min(self.base_population_factor * self.dim, self.population_size + 5) # Increased population adjustment
                    else:
                        self.mutation_factor = max(0.4, self.mutation_factor - 0.2)
                        self.crossover_rate = max(0.6, self.crossover_rate - 0.05)
                    
                if evaluations % (self.population_size * 4) == 0:
                    self.population = np.vstack((self.population, np.random.uniform(func.bounds.lb, func.bounds.ub, (1, self.dim))))

                # Reinitialize if diversity is too low
                if evaluations % (self.population_size * 10) == 0 and diversity < 0.05:
                    self.initialize_population(func.bounds.lb, func.bounds.ub)

        return min(self.population, key=func)