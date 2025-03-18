import numpy as np

class DualPhaseAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.noise_factor = 0.05
        self.layer_increment = max(1, dim // 5)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size
        phase_change = self.budget // 3

        while evaluations < self.budget:
            current_dim = min(self.dim, ((evaluations // self.layer_increment) + 1) * self.layer_increment)
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.7 - 0.1 * adaptive_factor
            cognitive_coeff = 2.0 if evaluations < phase_change else 1.5
            social_coeff = 1.7 if evaluations < phase_change else 2.0

            for i in range(self.population_size):
                r1, r2 = np.random.random(current_dim), np.random.random(current_dim)
                learning_rate = 0.4 + 0.05 * adaptive_factor
                self.velocity[i][:current_dim] = (inertia_weight * self.velocity[i][:current_dim] +
                                                  cognitive_coeff * r1 * (personal_best[i][:current_dim] - swarm[i][:current_dim]) +
                                                  social_coeff * r2 * (global_best[:current_dim] - swarm[i][:current_dim]))
                swarm[i][:current_dim] += learning_rate * self.velocity[i][:current_dim]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Differential evolution-like mutation
                mutant_idx = np.random.choice(self.population_size, 3, replace=False)
                mutant_vector = swarm[mutant_idx[0]] + 0.8 * (swarm[mutant_idx[1]] - swarm[mutant_idx[2]])
                mutant_vector = np.clip(mutant_vector, lb, ub)
                trial_vector = np.where(np.random.rand(self.dim) < 0.5, mutant_vector, swarm[i])

                noise = np.random.normal(0, self.noise_factor * (adaptive_factor ** 1.3), current_dim)
                trial_vector[:current_dim] += noise
                trial_vector = np.clip(trial_vector, lb, ub)

                f_value = func(trial_vector)
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = trial_vector
                    personal_best_value[i] = f_value

                if f_value < global_best_value:
                    global_best = trial_vector
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value