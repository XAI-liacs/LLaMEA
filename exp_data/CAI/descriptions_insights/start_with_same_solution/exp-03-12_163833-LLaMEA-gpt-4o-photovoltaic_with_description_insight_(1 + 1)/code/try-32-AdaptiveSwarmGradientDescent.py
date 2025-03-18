import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.swarm_clusters = 2 + int(np.log2(dim))  # New: Number of swarm clusters

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
            inertia_weight = 0.9 * adaptive_factor + 0.1  # Modified inertia weight
            cognitive_coeff = 1.5 + 0.5 * adaptive_factor  # Modified cognitive coefficient
            social_coeff = 2.0 - 0.5 * adaptive_factor  # Modified social coefficient

            dynamic_lr = 0.1 + 0.4 * adaptive_factor  # Modified dynamic learning rate

            velocity_scale = 1 + 0.3 * (1 - adaptive_factor)**1.5  # Adjusted scaling

            # New stochasticity and local exploration
            cluster_indices = np.random.choice(self.swarm_clusters, self.population_size)
            stochastic_factor = np.random.uniform(0.9, 1.3, self.swarm_clusters)

            for i in range(self.population_size):
                cluster_index = cluster_indices[i]
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += velocity_scale * dynamic_lr * self.velocity[i] * stochastic_factor[cluster_index]
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