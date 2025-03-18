import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_factor = 12  # Change 1: Adjusted base population factor
        self.population_size = self.base_population_factor * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.population = None
        self.stagnation_counter = 0  # Change 2: Introduced stagnation counter

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))

    def mutate(self, target_idx, lb, ub):
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
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
        alpha = 1 / (1 + np.exp(trial_score - target_score))
        if trial_score < target_score:
            self.stagnation_counter = 0  # Reset stagnation counter
            return trial_vector
        else:
            self.stagnation_counter += 1  # Increment stagnation counter
            return (alpha * trial_vector + (1 - alpha) * target_vector)

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
                        self.mutation_factor = min(1.0, self.mutation_factor + 0.1)  # Change 3: Adjusted mutation increment
                        self.crossover_rate = min(1.0, self.crossover_rate + 0.03)  # Change 4: Adjusted crossover increment
                    else:
                        self.mutation_factor = max(0.3, self.mutation_factor - 0.15)  # Change 5: Adjusted mutation decrement
                        self.crossover_rate = max(0.6, self.crossover_rate - 0.03)  # Change 6: Adjusted crossover decrement

                if evaluations % (self.population_size * 5) == 0:
                    random_vectors = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.dim, self.dim))
                    self.population[:self.dim] = random_vectors

                if self.stagnation_counter > (self.population_size * 0.5):  # Change 7: Stagnation-based strategy
                    self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
                    self.stagnation_counter = 0  # Reset stagnation counter after boost

        return min(self.population, key=func)