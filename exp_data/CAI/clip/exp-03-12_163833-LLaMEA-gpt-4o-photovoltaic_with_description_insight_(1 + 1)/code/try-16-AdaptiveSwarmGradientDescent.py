import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.learning_rates = np.full((self.population_size, dim), 0.1)  # Layer-wise learning rates

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
            inertia_weight = 0.9 * adaptive_factor + 0.1
            cognitive_coeff = 1.5 * adaptive_factor
            social_coeff = 1.5

            velocity_scale = 1 + 0.5 * (1 - adaptive_factor)**2

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += velocity_scale * self.learning_rates[i] * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Adaptive local search
                if evaluations % 10 == 0:
                    local_search = swarm[i] + 0.01 * np.random.normal(size=self.dim)
                    local_search = np.clip(local_search, lb, ub)
                    local_f_value = func(local_search)
                    evaluations += 1
                    if local_f_value < personal_best_value[i]:
                        swarm[i] = local_search
                        f_value = local_f_value
                    else:
                        f_value = func(swarm[i])
                        evaluations += 1
                else:
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