import numpy as np

class PSO_SA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 10 * dim
        self.inertia_weight = 0.7
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.temperature = 1.0
        self.cooling_rate = 0.98

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best_positions)
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        eval_count = self.swarm_size

        while eval_count < self.budget:
            for i in range(self.swarm_size):
                if eval_count >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_const * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_const * r2 * (global_best_position - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                fitness = func(positions[i])
                eval_count += 1

                if fitness < personal_best_fitness[i] or np.random.rand() < np.exp((personal_best_fitness[i] - fitness) / self.temperature):
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = fitness

                    if fitness < global_best_fitness:
                        global_best_position = positions[i]
                        global_best_fitness = fitness

            self.temperature *= self.cooling_rate

        return global_best_position