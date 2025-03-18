import numpy as np

class EnhancedAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.mutation_rate = 0.15
        self.layer_increase_step = np.ceil(dim / 8).astype(int)
        self.quasi_random_seed = np.random.rand(dim)  # New line

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
            inertia_weight = 0.6 + 0.4 * adaptive_factor
            cognitive_coeff = 1.5 * (1 + 0.5 * adaptive_factor)
            social_coeff = 1.5 * (1 - 0.5 * adaptive_factor)

            if evaluations % (self.budget // 10) == 0:  # New block
                self.population_size = min(self.population_size + 1, int(np.sqrt(self.dim) * 5))  # New block
                swarm = np.vstack((swarm, np.random.uniform(lb, ub, (1, self.dim))))  # New block
                self.velocity = np.vstack((self.velocity, np.zeros((1, self.dim))))  # New block

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                layer_scaling = 1 + 0.1 * (i % self.layer_increase_step) / self.layer_increase_step
                adaptive_scaling = (0.9 + 0.1 * np.sin(evaluations / self.budget * np.pi)) * layer_scaling
                self.velocity[i] = (adaptive_scaling * inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                if np.random.rand() < self.mutation_rate:
                    mutation_scale = np.random.normal(0, 0.3 * adaptive_scaling, self.dim)
                    swarm[i] += mutation_scale
                    swarm[i] = np.clip(swarm[i], lb, ub)

                # Quasi-random perturbation for diversity
                quasi_random_perturbation = 0.01 * (np.sin(self.quasi_random_seed * evaluations) - 0.5)  # New line
                swarm[i] += quasi_random_perturbation  # New line
                swarm[i] = np.clip(swarm[i], lb, ub)  # New line

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