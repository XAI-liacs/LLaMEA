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

        # Constraint on function evaluations
        evaluations = self.population_size
        stagnation_counter = 0

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.5 + 0.4 * adaptive_factor  # Adjusted dynamic inertia weight
            cognitive_coeff = 1.5 * (0.5 + 0.5 * adaptive_factor)
            social_coeff = 1.5 * (0.5 + 0.5 * adaptive_factor)  # Dynamic social coefficient
            
            adaptive_lr = 0.1 + 0.9 * adaptive_factor

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += adaptive_lr * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i] or np.random.rand() < 0.2 * adaptive_factor:  # Modified personal best update criteria
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                if f_value < global_best_value and np.random.rand() < 0.5 + 0.5 * adaptive_factor:
                    global_best = swarm[i]
                    global_best_value = f_value
                    stagnation_counter = 0  # Reset stagnation counter
                else:
                    stagnation_counter += 1

                if evaluations >= self.budget or stagnation_counter > 100:
                    break

        return global_best, global_best_value