import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))

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
            inertia_weight = 0.9 - 0.5 * (np.cos(np.pi * evaluations / self.budget))
            cognitive_coeff = np.random.uniform(1.0, 2.0) * adaptive_factor
            social_coeff = 1.5 * adaptive_factor
            
            neighborhood_radius = (ub - lb) * (0.1 + 0.4 * (1 - evaluations / self.budget))  # New line

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)
                
                local_search = swarm[i] + np.random.uniform(-neighborhood_radius, neighborhood_radius, self.dim)  # New line
                local_search = np.clip(local_search, lb, ub)                                                      # New line

                f_value = func(local_search)  # Changed line
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = local_search  # Changed line
                    personal_best_value[i] = f_value

                if f_value < global_best_value:
                    global_best = local_search  # Changed line
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value