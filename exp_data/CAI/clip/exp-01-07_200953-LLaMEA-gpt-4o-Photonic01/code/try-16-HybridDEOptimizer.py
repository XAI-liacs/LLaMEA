import numpy as np

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.eval_count = 0

    def __call__(self, func):
        # Initialize parameters
        pop_size = 10 + int(0.2 * self.dim)
        F = 0.5  # Mutation factor
        CR = 0.9  # Crossover probability
        adapt_rate = 0.2  # Adjustment rate for F and CR

        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (pop_size, self.dim))

        # Evaluate initial population
        fitness = np.apply_along_axis(func, 1, population)
        self.eval_count += pop_size

        best_idx = np.argmin(fitness)
        global_best_position = population[best_idx].copy()
        global_best_value = fitness[best_idx]

        while self.eval_count < self.budget:
            for i in range(pop_size):
                # Mutation
                indices = np.random.choice(np.delete(np.arange(pop_size), i), 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant_vector = x1 + F * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                # Crossover
                trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, population[i])

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                self.eval_count += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    # Update global best
                    if trial_fitness < global_best_value:
                        global_best_position = trial_vector
                        global_best_value = trial_fitness

                # Dynamic parameter adaptation
                if np.random.rand() < adapt_rate:
                    F = 0.4 + 0.5 * np.random.rand()
                    CR = 0.8 + 0.2 * np.random.rand()

                # Opposition-Based Learning
                if np.random.rand() < 0.1:
                    opposite_vector = lb + ub - population[i]
                    opposite_fitness = func(opposite_vector)
                    self.eval_count += 1
                    if opposite_fitness < fitness[i]:
                        population[i] = opposite_vector
                        fitness[i] = opposite_fitness
                        if opposite_fitness < global_best_value:
                            global_best_position = opposite_vector
                            global_best_value = opposite_fitness

                if self.eval_count >= self.budget:
                    break

        return global_best_position