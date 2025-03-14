import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.neighborhood_size = max(2, self.population_size // 5)

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
            inertia_weight = 0.9 - 0.5 * adaptive_factor
            cognitive_coeff = 2.0 * adaptive_factor  # Changed coefficient from 1.5 to 2.0
            social_coeff = 1.5 * np.exp(-0.5 * adaptive_factor)
            local_coeff = 0.4 * adaptive_factor

            # Introduce dynamic reduction in population size
            if evaluations / self.budget > 0.5:
                self.population_size = max(5, int(self.population_size * 0.9))
                self.velocity = self.velocity[:self.population_size]
                swarm = swarm[:self.population_size]
                personal_best = personal_best[:self.population_size]
                personal_best_value = personal_best_value[:self.population_size]

            for i in range(self.population_size):
                r1, r2, r3 = np.random.random(self.dim), np.random.random(self.dim), np.random.random(self.dim)
                neighborhood_indices = np.random.choice(self.population_size, self.neighborhood_size, replace=False)
                local_best = personal_best[neighborhood_indices[np.argmin(personal_best_value[neighborhood_indices])]]

                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) +
                                    local_coeff * r3 * (local_best - swarm[i]))
                swarm[i] += self.velocity[i]
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