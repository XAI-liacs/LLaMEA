import numpy as np

class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.temperature = 1.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_position = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        pop_fitness = np.array([func(ind) for ind in pop_position])

        evaluations = self.pop_size
        best_idx = np.argmin(pop_fitness)
        best_global_position = pop_position[best_idx]
        best_global_fitness = pop_fitness[best_idx]

        while evaluations < self.budget:
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop_position[np.random.choice(indices, 3, replace=False)]

                mutant_vector = np.clip(a + self.mutation_factor * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial_vector = np.where(crossover, mutant_vector, pop_position[i])

                trial_fitness = func(trial_vector)
                evaluations += 1

                if (trial_fitness < pop_fitness[i] or 
                    np.exp((pop_fitness[i] - trial_fitness) / self.temperature) > np.random.rand()):
                    pop_position[i] = trial_vector
                    pop_fitness[i] = trial_fitness

                if trial_fitness < best_global_fitness:
                    best_global_fitness = trial_fitness
                    best_global_position = trial_vector

                if evaluations >= self.budget:
                    break

            # Adaptive temperature reduction
            self.temperature *= 0.99

        return best_global_position, best_global_fitness