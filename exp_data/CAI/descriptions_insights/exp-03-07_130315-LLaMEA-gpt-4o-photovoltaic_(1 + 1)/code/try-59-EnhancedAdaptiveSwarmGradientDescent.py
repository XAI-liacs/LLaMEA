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
            inertia_weight = 0.8 * (0.9 - adaptive_factor)  # Modified line
            cognitive_coeff = 1.8 * adaptive_factor  # Modified line
            social_coeff = 1.6 + 0.2 * adaptive_factor  # Modified line

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                
                neighborhood_indices = np.random.choice(self.population_size, size=3, replace=False)  # Modified line
                neighborhood_best = personal_best[neighborhood_indices[np.argmin(personal_best_value[neighborhood_indices])]]  # Modified line
                fitness_based_influence = np.tanh(personal_best_value[i] - global_best_value) * np.random.random(self.dim)  # Modified line
                
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) +
                                    0.7 * np.random.random(self.dim) * (neighborhood_best - swarm[i]) +
                                    fitness_based_influence)  # Modified line
                velocity_scale = 0.6 + 0.4 * np.random.random()  # Modified line
                swarm[i] += velocity_scale * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = 0.95 * swarm[i] + 0.05 * personal_best[i]
                    personal_best_value[i] = f_value

                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value