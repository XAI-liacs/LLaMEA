import numpy as np

class EnhancedAdaptiveSwarmGradientDescent:
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

        evaluations = self.initial_population_size
        mutation_factor = 0.8

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.9 - 0.5 * adaptive_factor**3
            adaptive_velocity_factor = 0.5 + 0.5 * np.exp(-3 * adaptive_factor)
            cognitive_coeff = 1.5 * adaptive_factor * (1 + 0.2 * np.cos(evaluations * np.pi / 4))
            social_coeff = 1.5
            mutation_factor = 0.6 + 0.4 * adaptive_factor

            # Dynamic resizing of the swarm
            dynamic_population_size = self.initial_population_size + int(adaptive_factor * self.initial_population_size / 2)
            swarm = swarm[:dynamic_population_size]
            personal_best = personal_best[:dynamic_population_size]
            personal_best_value = personal_best_value[:dynamic_population_size]
            self.velocity = self.velocity[:dynamic_population_size]

            for i in range(dynamic_population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                indices = np.random.choice(dynamic_population_size, 3, replace=False)
                mutant_vector = (swarm[indices[0]] + mutation_factor * (swarm[indices[1]] - swarm[indices[2]]))
                mutant_vector = np.clip(mutant_vector, lb, ub)

                self.velocity[i] = (adaptive_velocity_factor * inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) +
                                    0.1 * (mutant_vector - swarm[i]))

                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                if f_value < global_best_value:
                    global_best = swarm[i] + np.random.normal(0, 0.01 * adaptive_factor, self.dim)
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value