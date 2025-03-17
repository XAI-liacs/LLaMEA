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

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.7 + 0.5 * adaptive_factor * 0.98  # Self-adaptive strategy for faster reduction
            learning_factor = 1 + 0.1 * adaptive_factor  # Dynamic learning factor adjustment
            cognitive_coeff = 1.5 * adaptive_factor * learning_factor
            social_coeff = 1.5

            # Dynamic population size adjustment
            current_population_size = self.population_size + int(adaptive_factor * 5)

            for i in range(current_population_size):
                if i >= self.population_size:
                    # Adaptive mutation for new members with increased variability
                    new_member = np.random.uniform(lb, ub, self.dim)
                    self.velocity = np.random.uniform(-1.5, 1.5, (current_population_size, self.dim))
                else:
                    r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                    self.velocity[i] = (inertia_weight * self.velocity[i] +
                                        cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                        social_coeff * r2 * (global_best - swarm[i]))
                    swarm[i] += self.velocity[i] + np.random.normal(0, 0.1, self.dim)  # Adding Gaussian noise
                    swarm[i] = np.clip(swarm[i], lb, ub)

                    # Evaluate and update personal best
                    f_value = func(swarm[i])
                    evaluations += 1
                    if f_value < personal_best_value[i]:
                        personal_best[i] = swarm[i]
                        personal_best_value[i] = f_value

                    # Update global best
                    if f_value < global_best_value:
                        global_best = swarm[i]
                        global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value