import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.base_population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.base_population_size, dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evaluations = 0
        
        while evaluations < self.budget:
            # Dynamically adjust population size
            population_size = self.base_population_size + int((self.budget - evaluations) / (self.budget / 5))
            swarm = np.random.uniform(lb, ub, (population_size, self.dim))
            personal_best = swarm.copy()
            personal_best_value = np.array([func(x) for x in swarm])
            global_best = personal_best[np.argmin(personal_best_value)]
            global_best_value = np.min(personal_best_value)

            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.9 - 0.5 * (np.cos(np.pi * evaluations / self.budget))
            cognitive_coeff = np.random.uniform(1.0, 2.0) * adaptive_factor
            social_coeff = 1.5 + 0.5 * (1 - adaptive_factor)  # Enhanced line

            for i in range(population_size):  # Adjusted for dynamic size
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i % self.base_population_size] = (inertia_weight * self.velocity[i % self.base_population_size] +
                                                              cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                                              social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += self.velocity[i % self.base_population_size]
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