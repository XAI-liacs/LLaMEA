import numpy as np

class EnhancedHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)  # Adaptive population size
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.mutation_factor = 0.8  # Differential mutation factor
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.inertia_initial = 0.9
        self.inertia_final = 0.4

    def chaotic_initialization(self, lb, ub):
        # Chaotic logistic map for initialization
        chaotic_seq = np.zeros((self.pop_size, self.dim))
        chaotic_seq[0] = np.random.rand(self.dim)
        for i in range(1, self.pop_size):
            chaotic_seq[i] = 4 * chaotic_seq[i - 1] * (1 - chaotic_seq[i - 1])
        return lb + chaotic_seq * (ub - lb)

    def dynamic_inertia(self, evaluations):
        return self.inertia_final + (self.inertia_initial - self.inertia_final) * \
               ((self.budget - evaluations) / self.budget)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop_position = self.chaotic_initialization(lb, ub)
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

            inertia_weight = self.dynamic_inertia(evaluations)

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_velocity = self.c1 * r1 * (personal_best_positions[i] - pop_position[i])
                social_velocity = self.c2 * r2 * (self.best_global_position - pop_position[i])
                pop_velocity[i] = inertia_weight * pop_velocity[i] + cognitive_velocity + social_velocity

                # Adaptive differential mutation
                if np.random.rand() < 0.1:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    donor_vector = (pop_position[indices[0]] +
                                    self.mutation_factor * (pop_position[indices[1]] - pop_position[indices[2]]))
                    np.clip(donor_vector, lb, ub, out=donor_vector)
                    pop_position[i] = donor_vector
                else:
                    pop_position[i] += pop_velocity[i]
                    np.clip(pop_position[i], lb, ub, out=pop_position[i])

        return self.best_global_position, self.best_global_fitness