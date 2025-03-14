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

        # Constraint on function evaluations
        evaluations = self.population_size
        prev_global_best_value = global_best_value  # New line

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            # Temperature-based adaptive inertia weight
            temperature = 1 - (evaluations / self.budget)
            inertia_weight = 0.4 + 0.6 * (1 - temperature ** 2)  # Updated line
            
            # Dynamically adapt coefficients based on distance to global best
            dist_factor = np.linalg.norm(global_best - swarm.mean(axis=0)) / np.linalg.norm(ub - lb)
            cognitive_coeff = 1.5 * (0.5 + 0.5 * (1 - dist_factor))
            social_coeff = 1.5 * (1 - 0.3 * adaptive_factor)

            elite_index = np.argmin(personal_best_value)  # New line
            for i in range(self.population_size):
                if i == elite_index:  # New line
                    continue  # Elite preservation

                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 ** 2 * (global_best - swarm[i]))  # Modified line
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Introduce adaptive mutation
                mutation_rate = 0.1 * adaptive_factor * temperature
                if np.random.rand() < mutation_rate * (1 - global_best_value):  # Modified line
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    swarm[i] += mutation_vector

                # Evaluate and update personal best
                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                # Update global best
                if f_value < global_best_value:
                    prev_global_best_value = global_best_value  # New line
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

            # Dynamic swarm reduction
            if evaluations % (self.budget // 10) == 0:
                self.population_size = max(2, int(self.population_size * 0.9))
                swarm = swarm[:self.population_size]
                personal_best = personal_best[:self.population_size]
                personal_best_value = personal_best_value[:self.population_size]
                self.velocity = self.velocity[:self.population_size]

        return global_best, global_best_value