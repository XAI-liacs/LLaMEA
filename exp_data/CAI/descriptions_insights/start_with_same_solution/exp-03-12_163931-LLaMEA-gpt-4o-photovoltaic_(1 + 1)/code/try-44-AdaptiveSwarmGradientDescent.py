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
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.9 - 0.5 * adaptive_factor
            improvement_factor = np.exp(-global_best_value / (np.min(personal_best_value) + 1e-10))
            cognitive_coeff = 1.7 * adaptive_factor * improvement_factor + 0.3 * np.random.random()
            social_coeff = 1.7 * improvement_factor + 0.3 * np.random.random()

            repulsion_coeff = 0.5 * (1 - improvement_factor) * np.random.random()

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) -
                                    repulsion_coeff * (swarm[i] - global_best))

                swarm[i] += self.velocity[i]
                dynamic_lb = lb + (ub - lb) * 0.05 * (evaluations / self.budget)
                dynamic_ub = ub - (ub - lb) * 0.05 * (evaluations / self.budget)
                swarm[i] = np.clip(swarm[i], dynamic_lb, dynamic_ub)

                # Temporal local search
                if evaluations % (self.population_size // 2) == 0:  # Changed line 1
                    swarm[i] += 0.01 * np.random.normal(0, 1, self.dim)  # Changed line 2

                # Evaluate and update personal best
                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                if np.random.random() < 0.05:  # Stochastic perturbation - Changed line 3
                    swarm[i] = np.random.uniform(dynamic_lb, dynamic_ub, self.dim)  # Changed line 4

                # Update global best
                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value