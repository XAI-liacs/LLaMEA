import numpy as np

class EnhancedAdaptiveSwarmGradientDescentV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.mutation_rate = 0.15
        self.layer_increase_step = np.ceil(dim / 8).astype(int)

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
            diversity_factor = np.std(swarm, axis=0).mean() / (ub - lb).mean()
            inertia_weight = (0.5 + 0.5 * np.cos(adaptive_factor * np.pi)) * (1 + diversity_factor)  # Changed line 1
            cognitive_coeff = 2.0 * (1 + 0.5 * adaptive_factor)  # Changed line 2
            social_coeff = 2.0 * (1 - 0.5 * adaptive_factor)  # Changed line 3

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                layer_scaling = 1 + 0.1 * (i % self.layer_increase_step) / self.layer_increase_step
                adaptive_scaling = (0.9 + 0.1 * np.sin(evaluations / self.budget * np.pi)) * layer_scaling  # Changed line 4
                self.velocity[i] = (adaptive_scaling * inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Introduce mutation for diversity
                adjusted_mutation_rate = self.mutation_rate * (1 + diversity_factor)
                if np.random.rand() < adjusted_mutation_rate:
                    non_uniform_scale = np.linalg.norm(global_best - swarm[i]) / np.sqrt(self.dim)
                    mutation_scale = (np.random.normal(0, 0.2 * adaptive_scaling * non_uniform_scale *  # Changed line 5
                                    (1 - adaptive_factor), self.dim) * (diversity_factor + 0.1))
                    hybrid_mutation = 0.5 * mutation_scale + 0.5 * np.random.normal(0, 0.1, self.dim)  # Changed line 6
                    swarm[i] += hybrid_mutation  # Changed line 7
                    swarm[i] = np.clip(swarm[i], lb, ub)

                # Evaluate and update personal best
                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                # Update global best
                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value