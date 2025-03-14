import numpy as np

class SynergisticGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.noise_factor = 0.01  # Initial noise level
        self.layer_weights = np.linspace(0.8, 1.2, self.population_size)  # Layer weights for adaptation

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
            inertia_weight = 0.6 + 0.2 * adaptive_factor  # Adjusted to optimize exploration
            cognitive_coeff = 1.9 * adaptive_factor  # Balanced for better exploration
            social_coeff = 1.8  # Slightly higher to improve global learning

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                layer_adaptation = self.layer_weights[i]  # Enhanced layer adaptation
                learning_rate = 0.3 * adaptive_factor * layer_adaptation  # Increased for dynamic convergence
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += learning_rate * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                noise = np.random.normal(0, self.noise_factor * adaptive_factor, self.dim)  # Refined noise with decay
                swarm[i] += noise
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