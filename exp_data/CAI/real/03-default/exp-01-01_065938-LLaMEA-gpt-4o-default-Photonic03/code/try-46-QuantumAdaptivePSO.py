import numpy as np

class QuantumAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.w = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.max_velocity = 0.2
        self.adapt_rate = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        velocity = np.random.uniform(-self.max_velocity, self.max_velocity, (self.swarm_size, self.dim))
        position_quantum = np.random.uniform(0, 1, (self.swarm_size, self.dim))
        position = lb + (ub - lb) * np.sin(np.pi * position_quantum)
        personal_best_position = np.copy(position)
        personal_best_fitness = np.array([func(x) for x in position])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_position[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity[i] = (self.w * velocity[i] +
                               self.c1 * r1 * (personal_best_position[i] - position[i]) +
                               self.c2 * r2 * (global_best_position - position[i]))
                velocity[i] = np.clip(velocity[i], -self.max_velocity, self.max_velocity)
                position[i] = position[i] + velocity[i]
                position[i] = np.clip(position[i], lb, ub)

                current_fitness = func(position[i])
                evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = current_fitness
                    personal_best_position[i] = position[i]

                    if current_fitness < global_best_fitness:
                        global_best_fitness = current_fitness
                        global_best_position = position[i]

            self.w = np.clip(self.w + self.adapt_rate * (np.random.rand() - 0.5), 0.4, 0.9)
            self.c1 = np.clip(self.c1 + self.adapt_rate * (np.random.rand() - 0.5), 1.0, 2.0)
            self.c2 = np.clip(self.c2 + self.adapt_rate * (np.random.rand() - 0.5), 1.0, 2.0)
            self.max_velocity = np.clip(self.max_velocity + self.adapt_rate * (np.random.rand() - 0.5), 0.1, 0.3)

        return global_best_position