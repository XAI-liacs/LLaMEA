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
            temp_factor = 1 - (evaluations / self.budget) ** 1.7  # Changed decay strategy slightly
            randomization_factor = np.random.uniform(0.85, 1.15)  # Adjusted randomization range
            inertia_weight = randomization_factor * (0.4 + 0.6 * np.cos((np.pi/2) * temp_factor))  # Adjusted inertia weight range
            decay_factor = temp_factor
            cognitive_coeff = decay_factor * 1.8 * np.random.uniform(0.5, 1.5) * (1 + 0.2 * np.cos(evaluations / self.budget * np.pi))  # Modified cognitive coefficient slightly
            social_coeff = decay_factor * 1.2 * np.random.uniform(0.5, 1.5)  # Adjusted social coefficient slightly

            for i in range(self.population_size):
                if np.random.rand() < 0.05:  # Introduce probabilistic velocity reinitialization
                    self.velocity[i] = np.random.uniform(-1, 1, self.dim) * (ub - lb)
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = inertia_weight * (self.velocity[i] +
                                    max(cognitive_coeff, 1.2 * decay_factor) * r1 * (personal_best[i] - swarm[i]) + 
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