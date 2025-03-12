import numpy as np

class EnhancedAdaptiveSwarmGradientDescent:
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
            inertia_weight = 0.6 + 0.1 * np.cos(np.pi * evaluations / self.budget)  # Changed line 1
            cognitive_coeff = 2.0 * adaptive_factor  # Changed line 2
            social_coeff = 1.8 + 0.2 * adaptive_factor  # Changed line 3

            for i in range(self.population_size):
                r1, r2, r3 = np.random.random(self.dim), np.random.random(self.dim), np.random.random(self.dim)  # Changed line 4
                neighborhood_best = personal_best[np.random.randint(self.population_size)]
                localized_influence = np.random.uniform(-0.05, 0.05, self.dim)
                random_vector = np.random.uniform(-1, 1, self.dim)  # Changed line 5
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) +
                                    0.6 * r3 * (neighborhood_best - swarm[i]) +  # Changed line 6
                                    localized_influence +
                                    0.1 * random_vector)  # Changed line 7
                velocity_scale = 0.4 + 0.6 * np.random.random()  # Changed line 8
                swarm[i] += velocity_scale * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = 0.90 * swarm[i] + 0.10 * personal_best[i]  # Changed line 9
                    personal_best_value[i] = f_value

                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value