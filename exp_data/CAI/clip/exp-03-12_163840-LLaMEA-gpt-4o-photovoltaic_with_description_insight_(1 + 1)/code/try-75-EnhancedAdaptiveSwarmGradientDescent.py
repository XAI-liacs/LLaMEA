import numpy as np

class EnhancedAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.mutation_rate = 0.15
        self.layer_increase_step = np.ceil(dim / 10).astype(int)  # Changed line

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
            inertia_weight = 0.5 + 0.4 * np.cos(adaptive_factor * np.pi)  # Changed line
            cognitive_coeff = 1.7 * (1 + 0.5 * adaptive_factor)  # Changed line
            social_coeff = 1.3 * (1 - 0.5 * adaptive_factor)  # Changed line

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                layer_scaling = 1 + 0.15 * (i % self.layer_increase_step) / self.layer_increase_step  # Changed line
                adaptive_scaling = (0.9 + 0.1 * np.sin(evaluations / self.budget * np.pi)) * layer_scaling  # Changed line
                self.velocity[i] = (adaptive_scaling * inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                diversity_factor = np.std(swarm, axis=0).mean() / (ub - lb).mean()
                adjusted_mutation_rate = self.mutation_rate * (1 + 0.8 * diversity_factor)  # Changed line
                if np.random.rand() < adjusted_mutation_rate:
                    non_uniform_scale = np.linalg.norm(global_best - swarm[i]) / np.sqrt(self.dim)
                    mutation_scale = np.random.normal(0, 0.3 * adaptive_scaling * non_uniform_scale, self.dim)  # Changed line
                    swarm[i] += mutation_scale
                    swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value