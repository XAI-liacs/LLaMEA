import numpy as np

class EnhancedAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.adaptive_lr = np.ones(self.population_size)
        self.memory = np.random.uniform(-0.1, 0.1, (self.population_size, dim))
        self.exploration_weight = np.random.uniform(0.2, 0.8)
        self.synergy_factor = 0.5 + np.random.uniform(0, 1)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.6 + 0.3 * np.sin(np.pi * evaluations / self.budget)
            cognitive_coeff = 1.5 * adaptive_factor * (self.synergy_factor + 0.1 * np.sin(2 * np.pi * evaluations / self.budget))
            social_coeff = 1.9 * adaptive_factor * (self.synergy_factor + 0.1 * np.sin(2 * np.pi * evaluations / self.budget))

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                stochastic_factor = np.random.uniform(-0.1, 0.1, self.dim)
                dynamic_adjustment = np.random.uniform(0.7, 1.3)
                improvement_rate = (global_best_value - np.min(personal_best_value)) / (1 + evaluations)
                velocity_magnitude = np.linalg.norm(self.velocity[i])  # Magnitude of velocity
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) +
                                    self.exploration_weight * stochastic_factor * (improvement_rate / (1 + velocity_magnitude)) + 
                                    0.05 * self.memory[i]) * dynamic_adjustment
                self.memory[i] = self.velocity[i]
                swarm[i] += self.adaptive_lr[i] * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value
                    self.adaptive_lr[i] *= 1.2
                else:
                    self.adaptive_lr[i] *= 0.8

                if velocity_magnitude > 1.0:  # Adjust learning rate based on velocity magnitude
                    self.adaptive_lr[i] *= 0.95

                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value