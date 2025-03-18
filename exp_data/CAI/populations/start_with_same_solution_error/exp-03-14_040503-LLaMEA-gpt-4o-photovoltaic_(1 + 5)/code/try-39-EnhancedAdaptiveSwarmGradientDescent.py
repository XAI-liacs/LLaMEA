import numpy as np

class EnhancedAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.lb = None
        self.ub = None

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        def nonlinear_inertia_weight(t, T):
            return 0.9 - (0.5 * (t / T) ** 2)

        def diversity_metric(swarm):
            return np.mean(np.std(swarm, axis=0))

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = nonlinear_inertia_weight(evaluations, self.budget)
            cognitive_coeff = 2.0 * (0.5 + 0.5 * adaptive_factor**2)  # Modified line
            social_coeff = 2.0 * (1.1 - adaptive_factor**2)  # Modified line

            current_population_size = self.population_size - int(0.15 * self.population_size * evaluations / self.budget)

            for i in range(current_population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))

                self.velocity[i] = np.clip(self.velocity[i], -0.2 * (self.ub - self.lb), 0.2 * (self.ub - self.lb))

                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], self.lb, self.ub)

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
            
            if diversity_metric(swarm) < 0.01:
                num_reintroduce = max(1, int(0.1 * current_population_size))
                for _ in range(num_reintroduce):
                    idx = np.random.randint(current_population_size)
                    swarm[idx] = np.random.uniform(self.lb, self.ub, self.dim)

        return global_best, global_best_value