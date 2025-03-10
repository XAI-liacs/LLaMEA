import numpy as np

class EnhancedAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.adaptive_threshold = 0.1  # New line
        self.grad_step_size = 0.01  # New line

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
            inertia_weight = 0.5 + 0.4 * adaptive_factor  # Changed line
            cognitive_coeff = 1.5 * adaptive_factor  # Changed line
            social_coeff = 1.6 + 0.3 * adaptive_factor  # Changed line

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                
                neighborhood_best = personal_best[np.random.randint(self.population_size)]
                localized_influence = np.random.uniform(-0.02, 0.02, self.dim)  # Changed line
               
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) +
                                    0.5 * np.random.random(self.dim) * (neighborhood_best - swarm[i]))  # Changed line

                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]  # Changed line
                    personal_best_value[i] = f_value

                # Gradient-based local search step (New lines)
                if np.random.random() < self.adaptive_threshold:
                    gradient_step = np.clip(self.grad_step_size * np.sign(personal_best[i] - global_best), lb, ub)
                    swarm[i] = np.clip(swarm[i] - gradient_step, lb, ub)

                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value