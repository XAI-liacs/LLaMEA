import numpy as np

class AdaptiveSwarmGradientDescentEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.local_search_intensity = 5  # New parameter for local search

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
            inertia_weight = 0.5 + 0.3 * np.cos(evaluations / self.budget * np.pi)  # Modified line
            cognitive_coeff = 1.5 * adaptive_factor  # Modified line
            social_coeff = 1.5 + 0.5 * (1 - adaptive_factor)  # Modified line

            cooling_schedule = np.exp(-(evaluations / self.budget)**2)
            
            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                max_velocity = (ub - lb) * 0.2 * cooling_schedule  # Modified line
                self.velocity[i] = np.clip(self.velocity[i], -max_velocity, max_velocity)
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Local search enhancement
                for _ in range(self.local_search_intensity):  # New loop for local search
                    local_candidate = swarm[i] + np.random.normal(0, 0.1, self.dim)
                    local_candidate = np.clip(local_candidate, lb, ub)
                    local_value = func(local_candidate)
                    evaluations += 1
                    if local_value < personal_best_value[i]:
                        personal_best[i] = local_candidate
                        personal_best_value[i] = local_value
                    if local_value < global_best_value:
                        global_best = local_candidate
                        global_best_value = local_value

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