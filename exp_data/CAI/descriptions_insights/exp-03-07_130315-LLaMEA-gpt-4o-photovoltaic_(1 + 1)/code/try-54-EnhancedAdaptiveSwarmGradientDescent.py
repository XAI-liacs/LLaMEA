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

        # Constraint on function evaluations
        evaluations = self.population_size

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.5 + 0.4 * np.random.rand()  # Modified line
            cognitive_coeff = 1.5 + 0.5 * np.random.rand()  # Modified line
            social_coeff = 1.5 + 0.5 * adaptive_factor

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                
                neighborhood_best = personal_best[np.random.randint(self.population_size)]
                localized_influence = np.random.uniform(-0.05, 0.15, self.dim)  # Modified line
                velocity_mutation = 0.1 * np.random.randn(self.dim)  # New line
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) +
                                    0.5 * np.random.random(self.dim) * (neighborhood_best - swarm[i]) +  # Modified line
                                    localized_influence + velocity_mutation)  # Modified line
                velocity_scale = 0.5 + 0.5 * np.random.random()
                swarm[i] += velocity_scale * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Evaluate and update personal best
                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = 0.9 * swarm[i] + 0.1 * personal_best[i]  # Modified line
                    personal_best_value[i] = f_value

                # Update global best
                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value