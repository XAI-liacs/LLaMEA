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
            temp_factor = 1 - (evaluations / self.budget) ** 1.8  # Adjusted decay rate
            inertia_weight = 0.4 + 0.5 * np.sin(np.pi * temp_factor)  # Sine-based inertia adjustment
            decay_factor = temp_factor
            cognitive_coeff = decay_factor * 1.9 * np.random.uniform(0.4, 1.6) * (1 + 0.25 * np.sin(evaluations / self.budget * np.pi))  # Modified cognitive coefficient
            social_coeff = decay_factor * 1.5 * np.random.uniform(0.4, 1.6) * (1 + 0.15 * np.sin(evaluations / self.budget * np.pi))  # Modified social coefficient

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = inertia_weight * self.velocity[i] + cognitive_coeff * r1 * (personal_best[i] - swarm[i]) + social_coeff * r2 * (global_best - swarm[i])
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