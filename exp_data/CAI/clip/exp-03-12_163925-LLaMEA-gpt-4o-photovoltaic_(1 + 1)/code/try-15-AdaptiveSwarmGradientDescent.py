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
            inertia_weight = 0.9 - 0.5 * adaptive_factor  # Adjusted inertia weight strategy
            cognitive_coeff = 1.5 * adaptive_factor
            social_coeff = 1.5 + 0.5 * (1 - adaptive_factor)  # Make social coefficient slightly dynamic

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                dampening_factor = 0.95 + 0.05 * adaptive_factor  # Introduced adaptive velocity dampening factor
                
                # Line changed: Introduced velocity normalization
                self.velocity[i] = dampening_factor * (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                self.velocity[i] = self.velocity[i] / np.linalg.norm(self.velocity[i])  # Normalize velocity
                
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Adaptive mutation based on convergence
                mutation_rate = 0.02 + 0.1 * (1 - adaptive_factor) * (0.5 + 0.5 * adaptive_factor)  # Added decay function
                mutation_strength = 0.1 * (1 - adaptive_factor) * (0.5 + 0.5 * adaptive_factor)  # Added decay function
                if np.random.rand() < mutation_rate:  
                    swarm[i] += np.random.normal(0, mutation_strength, self.dim)
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