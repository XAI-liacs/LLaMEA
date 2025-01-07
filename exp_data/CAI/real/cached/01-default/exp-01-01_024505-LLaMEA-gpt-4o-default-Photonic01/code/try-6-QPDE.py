import numpy as np

class QPDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)  # Adaptive population size
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.mutation_factor = 0.8  # Differential mutation factor
        self.best_global_position = None
        self.best_global_fitness = float('inf')

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_position = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        pop_velocity = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.pop_size, self.dim))
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

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - pop_position[i])
                social_velocity = self.c2 * r2 * (self.best_global_position - pop_position[i])
                pop_velocity[i] = self.w * pop_velocity[i] + cognitive_velocity + social_velocity

                # Quantum behavior
                if np.random.rand() < 0.3:
                    theta = np.random.rand(self.dim) * (ub - lb) + lb
                    quantum_position = self.best_global_position + 0.5 * (np.sin(theta) - 0.5)
                    np.clip(quantum_position, lb, ub, out=quantum_position)
                    pop_position[i] = quantum_position
                else:
                    # Differential mutation
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    donor_vector = (pop_position[indices[0]] +
                                    self.mutation_factor * (pop_position[indices[1]] - pop_position[indices[2]]))
                    np.clip(donor_vector, lb, ub, out=donor_vector)
                    if np.random.rand() < 0.5:
                        pop_position[i] = donor_vector
                    else:
                        pop_position[i] += pop_velocity[i]
                        np.clip(pop_position[i], lb, ub, out=pop_position[i])

        return self.best_global_position, self.best_global_fitness