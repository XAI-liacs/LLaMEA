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
        prev_global_best_value = global_best_value

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            improvement_rate = np.abs(prev_global_best_value - global_best_value) / (np.abs(prev_global_best_value) + 1e-9)
            inertia_weight = 0.5 + 0.5 * improvement_rate
            
            # Dynamic learning rate based on the progress toward global best
            dist_factor = np.linalg.norm(global_best - swarm.mean(axis=0)) / np.linalg.norm(ub - lb)
            cognitive_coeff = 1.5 * (0.5 + 0.5 * (1 - dist_factor)) * (1 + adaptive_factor)  # Updated line
            social_coeff = 1.5 * (1 - 0.3 * adaptive_factor) * (1 + improvement_rate)  # Updated line

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                temperature = 1 - (evaluations / self.budget)
                mutation_rate = 0.1 * adaptive_factor * temperature
                if np.random.rand() < mutation_rate:
                    mutation_vector = np.random.normal(0, 0.1, self.dim) * (1 - dist_factor)  # Updated line
                    swarm[i] += mutation_vector

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                if f_value < global_best_value:
                    prev_global_best_value = global_best_value
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

            if evaluations % (self.budget // 10) == 0:
                self.population_size = max(2, int(self.population_size * 0.9))
                swarm = swarm[:self.population_size]
                personal_best = personal_best[:self.population_size]
                personal_best_value = personal_best_value[:self.population_size]
                self.velocity = self.velocity[:self.population_size]

        return global_best, global_best_value