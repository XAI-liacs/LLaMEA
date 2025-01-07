import numpy as np

class Adaptive_Levy_Flight_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 25
        self.alpha = 0.9  # Influence of local best
        self.beta = 0.3   # Influence of global best
        self.mutation_strength = 0.01
        self.inertia_weight = 0.7
        self.inertia_decay = 0.95  # Decaying inertia
        self.levy_scale = 0.1  # Scale factor for Levy flight

    def levy(self):
        # Generating step from Levy distribution
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / (np.abs(v) ** (1 / 3))
        return step * self.levy_scale

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_pos = np.copy(position)
        personal_best_val = np.array([func(p) for p in personal_best_pos])
        global_best_pos = personal_best_pos[np.argmin(personal_best_val)]
        global_best_val = np.min(personal_best_val)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                self.inertia_weight *= self.inertia_decay
                velocity[i] = (self.inertia_weight * velocity[i] +
                               self.alpha * r1 * (personal_best_pos[i] - position[i]) +
                               self.beta * r2 * (global_best_pos - position[i]))

                position[i] += velocity[i]
                # Levy flight perturbation
                position[i] += self.levy()
                position[i] = np.clip(position[i], lb, ub)

                current_val = func(position[i])
                evaluations += 1

                if current_val < personal_best_val[i]:
                    personal_best_pos[i] = position[i]
                    personal_best_val[i] = current_val

                if current_val < global_best_val:
                    global_best_pos = position[i]
                    global_best_val = current_val

                if evaluations >= self.budget:
                    break

        return global_best_pos, global_best_val