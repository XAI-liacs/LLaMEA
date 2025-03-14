import numpy as np

class AdaptiveSwarmGradientDescentEnhanced:
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
            inertia_weight = 0.5 + 0.2 * np.cos(evaluations / self.budget * np.pi) # Modified line
            cognitive_coeff = 1.5 * adaptive_factor # Modified line
            social_coeff = 1.8 + 0.2 * (1 - adaptive_factor) # Modified line

            cooling_schedule = np.exp(-(evaluations / self.budget)**2)
            
            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                max_velocity = (ub - lb) * 0.1 * cooling_schedule
                self.velocity[i] = np.clip(self.velocity[i], -max_velocity, max_velocity) # Modified line
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
                    # Hybrid local search: Apply a simple hill climbing step (Added section)
                    neighbor = global_best + np.random.uniform(-0.01, 0.01, self.dim)
                    neighbor = np.clip(neighbor, lb, ub)
                    neighbor_value = func(neighbor)
                    evaluations += 1
                    if neighbor_value < global_best_value:
                        global_best = neighbor
                        global_best_value = neighbor_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value