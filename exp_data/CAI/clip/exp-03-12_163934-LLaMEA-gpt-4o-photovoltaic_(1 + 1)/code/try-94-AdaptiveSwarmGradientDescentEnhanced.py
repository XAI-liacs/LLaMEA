import numpy as np

class AdaptiveSwarmGradientDescentEnhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.initial_population_size = self.population_size  # New: Initial population size
        self.min_population_size = 5  # New: Minimum population size

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
            inertia_weight = 0.7 * np.exp(-0.5 * (evaluations / self.budget))  # Changed: Nonlinear decay
            cognitive_coeff = 1.7 * adaptive_factor
            social_coeff = 1.7 + 0.3 * (1 - adaptive_factor)

            cooling_schedule = np.exp(-(evaluations / self.budget)**2)
            
            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                max_velocity = (ub - lb) * 0.1 * cooling_schedule
                self.velocity[i] = np.clip(self.velocity[i], -max_velocity * adaptive_factor, max_velocity * adaptive_factor)
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

            # New: Adaptive population size
            if evaluations % 50 == 0:  # Change population size every 50 evaluations
                self.population_size = max(self.min_population_size, int(self.initial_population_size * adaptive_factor))
                swarm = np.resize(swarm, (self.population_size, self.dim))
                personal_best = np.resize(personal_best, (self.population_size, self.dim))
                personal_best_value = np.resize(personal_best_value, self.population_size)
                self.velocity = np.resize(self.velocity, (self.population_size, self.dim))

        return global_best, global_best_value