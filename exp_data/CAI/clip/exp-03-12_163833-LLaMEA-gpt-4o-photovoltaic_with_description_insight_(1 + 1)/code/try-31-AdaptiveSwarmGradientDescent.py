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
            convergence_factor = np.log10(evaluations + 10) / np.log10(self.budget)  # New convergence-enhancing factor
            inertia_weight = (0.85 * adaptive_factor + 0.15) * convergence_factor  # Modified inertia weight
            cognitive_coeff = 1.6 * adaptive_factor
            social_coeff = 1.8

            # Introduce modified dynamic learning rate
            dynamic_lr = 0.05 + 0.45 * adaptive_factor

            # Adaptive velocity scaling based on historical convergence rate
            velocity_scale = 1 + 0.5 * (1 - adaptive_factor)**2
            # Stochastic velocity adjustment
            stochastic_factor = np.random.uniform(0.8, 1.2, self.population_size)

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += velocity_scale * dynamic_lr * self.velocity[i] * stochastic_factor[i]
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
                
                # Gradient approximation for local search enhancement
                if evaluations < self.budget:
                    grad_approx = (func(swarm[i] + 1e-5) - f_value) / 1e-5
                    swarm[i] -= 0.01 * grad_approx * (swarm[i] - global_best)
                    evaluations += 1

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value