import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.initial_population_size, dim))

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        # Constraints on function evaluations and initializations
        evaluations = self.initial_population_size
        previous_best_value = global_best_value

        while evaluations < self.budget:
            convergence_rate = np.abs(previous_best_value - global_best_value) / (previous_best_value + 1e-10)
            previous_best_value = global_best_value
            population_size = max(5, int(self.initial_population_size * (1 - convergence_rate)))
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.6 + 0.4 * np.power(adaptive_factor, 3)
            cognitive_coeff = 2.0 * adaptive_factor
            social_coeff = 1.3 + 0.2 * (np.max(personal_best_value) - np.min(personal_best_value)) / np.mean(personal_best_value)

            for i in range(population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
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