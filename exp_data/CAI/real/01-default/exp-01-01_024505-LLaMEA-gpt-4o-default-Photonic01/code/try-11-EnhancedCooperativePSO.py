import numpy as np

class EnhancedCooperativePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        
        # Initialize parameters for dynamic adaptation
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.4
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.8  # Factor for DE-based mutation
        self.crossover_rate = 0.9   # Crossover rate for DE

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        pop_position = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        personal_best_positions = np.copy(pop_position)
        personal_best_fitness = np.full(self.pop_size, float('inf'))

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.pop_size):
                fitness = func(pop_position[i])
                evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = pop_position[i]

                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_position = pop_position[i]

                if evaluations >= self.budget:
                    break

            inertia_weight = self.inertia_weight_max - (
                (self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget)
            )

            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Update velocity with dynamic inertia
                velocity[i] = (inertia_weight * velocity[i] +
                               self.cognitive_coeff * r1 * (personal_best_positions[i] - pop_position[i]) +
                               self.social_coeff * r2 * (self.best_global_position - pop_position[i]))

                # Apply DE-like mutation and crossover
                if np.random.rand() < self.crossover_rate:
                    a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                    mutant_vector = personal_best_positions[a] + self.mutation_factor * (personal_best_positions[b] - personal_best_positions[c])
                    mutant_vector = np.clip(mutant_vector, lb, ub)
                    pop_position[i] = np.where(np.random.rand(self.dim) < 0.5, mutant_vector, pop_position[i])
                else:
                    pop_position[i] = pop_position[i] + velocity[i]

                # Ensure positions are within bounds
                pop_position[i] = np.clip(pop_position[i], lb, ub)

        return self.best_global_position, self.best_global_fitness