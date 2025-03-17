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
            inertia_weight = 0.5 + 0.4 * np.sin(np.pi * adaptive_factor)  # Adjusted inertia weight
            cognitive_coeff = 1.5 * adaptive_factor
            social_coeff = 1.5

            learning_rate = 0.1 + 0.9 * adaptive_factor  # Adaptive learning rate

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += learning_rate * self.velocity[i]  # Applied adaptive learning rate
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

                # Adaptive mutation
                if np.random.rand() < 0.1:  # 10% chance
                    mutation = np.random.normal(0, 0.1, self.dim)
                    swarm[i] = np.clip(swarm[i] + mutation, lb, ub)

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value