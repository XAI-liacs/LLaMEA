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
        F = 0.8  # Differential evolution factor
        CR = 0.9  # Crossover probability

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            cognitive_coeff = np.random.uniform(1.0, 2.0) * adaptive_factor
            social_coeff = 1.5

            for i in range(self.population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                if np.random.rand() < CR:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    donor_vector = personal_best[indices[0]] + F * (personal_best[indices[1]] - personal_best[indices[2]])
                    trial_vector = np.where(np.random.rand(self.dim) < CR, donor_vector, swarm[i])
                else:
                    trial_vector = swarm[i] + cognitive_coeff * r1 * (personal_best[i] - swarm[i]) + social_coeff * r2 * (global_best - swarm[i])
                trial_vector = np.clip(trial_vector, lb, ub)

                # Evaluate and update personal best
                f_value = func(trial_vector)
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = trial_vector
                    personal_best_value[i] = f_value

                # Update global best
                if f_value < global_best_value:
                    global_best = trial_vector
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value