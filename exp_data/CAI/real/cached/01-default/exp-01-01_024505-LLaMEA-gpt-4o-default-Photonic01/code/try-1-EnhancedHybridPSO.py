import numpy as np

class EnhancedHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_factor = 0.8
        self.best_global_position = None
        self.best_global_fitness = float('inf')

    def levy_flight(self, L):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v)**(1 / beta)
        return L * step

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

                if np.random.rand() < 0.1:
                    indices = np.random.choice(self.pop_size, 3, replace=False)
                    donor_vector = (pop_position[indices[0]] +
                                    self.mutation_factor * (pop_position[indices[1]] - pop_position[indices[2]]))
                    np.clip(donor_vector, lb, ub, out=donor_vector)
                    pop_position[i] = donor_vector
                else:
                    pop_position[i] += pop_velocity[i]
                    np.clip(pop_position[i], lb, ub, out=pop_position[i])

                # Lévy flight step
                if np.random.rand() < 0.05:
                    levy_step = self.levy_flight(0.01 * (ub - lb))
                    pop_position[i] += levy_step
                    np.clip(pop_position[i], lb, ub, out=pop_position[i])

        return self.best_global_position, self.best_global_fitness