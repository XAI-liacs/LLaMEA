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
            inertia_weight = 0.5 + 0.2 * np.cos(evaluations / self.budget * np.pi) # Change 1
            cognitive_coeff = 1.5 * adaptive_factor # Change 2
            social_coeff = 1.5 + 0.5 * (1 - adaptive_factor) # Change 3

            # Adaptive cooling and perturbation strategy
            cooling_schedule = np.exp(-(evaluations / self.budget)**2) # Change 4

            perturbation_factor = 0.1 * (1 - adaptive_factor) # Change 5

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                max_velocity = (ub - lb) * 0.1 * cooling_schedule
                self.velocity[i] = np.clip(self.velocity[i], -max_velocity, max_velocity)
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Hybrid local search mechanism
                local_search_step = perturbation_factor * (2 * np.random.random(self.dim) - 1) # Change 6
                candidate_solution = swarm[i] + local_search_step # Change 7
                candidate_solution = np.clip(candidate_solution, lb, ub) # Change 8
                candidate_value = func(candidate_solution) # Change 9

                f_value = func(swarm[i])
                evaluations += 1

                if candidate_value < f_value: # Change 10
                    swarm[i] = candidate_solution # Change 11
                    f_value = candidate_value # Change 12

                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value