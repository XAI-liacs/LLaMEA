import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        evaluations = 0
        population_size = self.initial_population_size

        while evaluations < self.budget:
            # Dynamically adjust population size
            population_size = self.initial_population_size + int(0.1 * self.initial_population_size * (evaluations / self.budget))
            if self.velocity is None or self.velocity.shape[0] != population_size:
                self.velocity = np.zeros((population_size, self.dim))
            swarm = np.random.uniform(lb, ub, (population_size, self.dim))
            personal_best = swarm.copy()
            personal_best_value = np.array([func(x) for x in swarm])
            global_best = personal_best[np.argmin(personal_best_value)]
            global_best_value = np.min(personal_best_value)

            previous_best_value = global_best_value

            for i in range(population_size):
                adaptive_factor = 1 - evaluations / self.budget
                inertia_weight = 0.7 + 0.3 * adaptive_factor
                cognitive_coeff = 1.5 * adaptive_factor
                improvement_rate = (previous_best_value - global_best_value) / previous_best_value if previous_best_value != 0 else 0
                social_coeff = 1.5 * (1 + improvement_rate)

                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]))
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Evaluate and update personal best
                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                # Update global best
                if f_value < global_best_value:
                    previous_best_value = global_best_value
                    global_best = swarm[i]
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value