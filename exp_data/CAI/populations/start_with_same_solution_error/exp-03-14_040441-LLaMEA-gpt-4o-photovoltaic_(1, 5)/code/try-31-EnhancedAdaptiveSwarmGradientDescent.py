# Description: Enhance convergence by employing a dynamically adjusted swarm size and adaptive velocity scaling.
# Code:
import numpy as np

class EnhancedAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_size = 10 + 2 * int(np.sqrt(dim))
        self.population_size = self.base_size
        self.velocity = np.zeros((self.population_size, dim))
        self.adaptive_lr = np.ones(self.population_size)
        self.memory = np.random.uniform(-0.1, 0.1, (self.population_size, dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            self.population_size = self.base_size + int((self.budget - evaluations) / self.budget * self.base_size)
            inertia_weight = 0.8 - 0.5 * (evaluations / self.budget)
            cognitive_coeff = 2.0 * (1 - evaluations / self.budget)
            social_coeff = 2.0

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                stochastic_factor = np.random.uniform(-0.1, 0.1, self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) + 
                                    stochastic_factor)
                self.memory[i] = self.velocity[i]
                velocity_scale = np.clip(np.linalg.norm(self.velocity[i]), 0, 1.5)
                swarm[i] += self.adaptive_lr[i] * self.velocity[i] * velocity_scale
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value
                    self.adaptive_lr[i] = min(1.2, self.adaptive_lr[i] * 1.2)
                else:
                    self.adaptive_lr[i] *= 0.9

                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value