import numpy as np

class AdaptiveSwarmLevyFlight:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + 2 * int(np.sqrt(dim))
        self.population_size = self.initial_population_size
        self.velocity = np.zeros((self.population_size, dim))

    def levy_flight(self, size, dim):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=(size, dim))
        v = np.random.normal(0, 1, size=(size, dim))
        step = u / np.abs(v)**(1 / beta)
        return step

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
            inertia_weight = 0.5 + 0.4 * np.cos(2 * np.pi * evaluations / self.budget)
            cognitive_coeff = 1.5 * adaptive_factor
            social_coeff = 1.5 * (1 + 0.1 * adaptive_factor)

            if evaluations % 10 == 0:  # Dynamic population adjustment
                self.population_size = max(2, int(self.initial_population_size * (1 - 0.1 * adaptive_factor)))
                swarm = swarm[:self.population_size]
                personal_best = personal_best[:self.population_size]
                personal_best_value = personal_best_value[:self.population_size]
                self.velocity = self.velocity[:self.population_size]

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                damping_factor = 0.9 + 0.1 * (1 - adaptive_factor)
                self.velocity[i] = (damping_factor * inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                exploration_noise = self.levy_flight(1, self.dim).flatten() * adaptive_factor  # Lévy flight exploration
                swarm[i] += self.velocity[i] + exploration_noise
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