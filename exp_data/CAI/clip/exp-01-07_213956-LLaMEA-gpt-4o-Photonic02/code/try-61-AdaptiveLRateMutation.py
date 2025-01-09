import numpy as np

class AdaptiveLRateMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size
        self.min_lr = 0.1   # Minimum learning rate
        self.max_lr = 0.9   # Maximum learning rate
        self.CR = 0.9       # DE crossover probability
        self.mutation_factors = [0.5, 0.8, 1.2]  # Diverse mutation factors

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        personal_best = population.copy()
        personal_best_values = np.array([func(ind) for ind in personal_best])
        global_best = personal_best[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                lr = self.min_lr + (self.max_lr - self.min_lr) * (1 - evaluations / self.budget)
                candidate = population[i] + lr * (global_best - population[i])
                candidate = np.clip(candidate, lb, ub)

                # Diverse Differential Evolution mutation and crossover
                if np.random.rand() < self.CR:
                    idxs = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    F = np.random.choice(self.mutation_factors)  # Randomly select mutation factor
                    mutation_vector = a + F * (b - c)
                    mutation_vector = np.clip(mutation_vector, lb, ub)
                    crossover = np.random.rand(self.dim) < self.CR
                    candidate[crossover] = mutation_vector[crossover]

                candidate_value = func(candidate)
                evaluations += 1

                # Update personal and global bests
                if candidate_value < personal_best_values[i]:
                    personal_best[i] = candidate
                    personal_best_values[i] = candidate_value

                    if candidate_value < global_best_value:
                        global_best = candidate
                        global_best_value = candidate_value

                if evaluations >= self.budget:
                    break

        return global_best