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
        alpha = 1 / (1 + np.exp(trial_score - target_score))  # Dynamic weighting based on scores
        return trial_vector if trial_score < target_score else (alpha * trial_vector + (1 - alpha) * target_vector)

    def __call__(self, func):
        self.initialize_population(func.bounds.lb, func.bounds.ub)
        evaluations = 0
        success_count = 0  # Track successful trial vectors

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                mutant_vector = self.mutate(i, func.bounds.lb, func.bounds.ub)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                
                if func(trial_vector) < func(self.population[i]):
                    success_count += 1  # Increment success count if trial is better
                
                self.population[i] = self.selection(i, trial_vector, func)
                evaluations += 1

                if (evaluations % (self.population_size * 2)) == 0:
                    success_rate = success_count / (self.population_size * 2)
                    success_count = 0  # Reset success count
                    if success_rate > 0.2:
                        self.mutation_factor = min(1.0, self.mutation_factor + 0.05)
                        self.crossover_rate = min(1.0, self.crossover_rate + 0.05)
                    else:
                        self.mutation_factor = max(0.3, self.mutation_factor - 0.05)
                        self.crossover_rate = max(0.6, self.crossover_rate - 0.05)
                
                if evaluations % (self.population_size * 4) == 0:
                    self.population_size = min(self.base_population_factor * self.dim, self.population_size + 1)
                    self.population = np.vstack((self.population, np.random.uniform(func.bounds.lb, func.bounds.ub, (1, self.dim))))

                if evaluations % (self.population_size * 5) == 0:  # New strategic diversity boost
                    random_vectors = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim, self.dim))
                    self.population[:self.dim] = random_vectors

        return min(self.population, key=func)