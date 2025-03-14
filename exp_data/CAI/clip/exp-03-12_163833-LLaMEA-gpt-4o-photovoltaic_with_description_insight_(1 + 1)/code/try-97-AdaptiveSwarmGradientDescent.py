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
            convergence_factor = np.log10(evaluations + 10) / np.log10(self.budget)
            inertia_weight = (0.75 * adaptive_factor + 0.25) * convergence_factor * np.sin(2 * np.pi * evaluations / self.budget) 
            cognitive_coeff = 1.5 * (adaptive_factor ** 0.4)  # Dynamic cognitive coefficient
            social_coeff = 2.0  # Updated social coefficient

            dynamic_lr = 0.05 + 0.5 * adaptive_factor  # Refined dynamic learning rate
            velocity_scale = 1 + 0.5 * (1 - adaptive_factor)**1.75 * 0.85  # Adjusted velocity scaling
            stochastic_factor = np.random.uniform(0.9, 1.3, self.population_size)  # Adjusted stochastic range

            perturbation_factor = 0.08 * np.sin(3.0 * np.pi * evaluations / (self.budget * self.dim)) * adaptive_factor  # Further refined perturbation formula

            if evaluations > self.budget * 0.4:  # Fine-tuned dynamic layer count trigger
                self.dim = min(self.dim + 1, len(swarm[0]))

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += velocity_scale * dynamic_lr * self.velocity[i] * stochastic_factor[i] + perturbation_factor
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