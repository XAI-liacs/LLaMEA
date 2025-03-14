import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 3 * int(np.sqrt(dim))
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
            adaptive_factor = 1 - (evaluations / self.budget) ** 2
            inertia_weight = 0.5 + 0.1 * np.random.rand() * (1 + adaptive_factor)  # Modified line
            cognitive_coeff = 2.0 * (1 - adaptive_factor) * (1 + np.cos(evaluations / self.budget * np.pi))
            social_coeff = 1.5 + 0.5 * np.cos(evaluations / self.budget * np.pi)

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                neighbors = np.random.choice(self.population_size, size=int(5 + adaptive_factor * 4), replace=False)
                local_best = personal_best[neighbors[np.argmin(personal_best_value[neighbors])]]
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (local_best - swarm[i]))
                velocity_clamp = 0.1 * (ub - lb)
                self.velocity[i] = np.clip(self.velocity[i], -velocity_clamp, velocity_clamp)
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