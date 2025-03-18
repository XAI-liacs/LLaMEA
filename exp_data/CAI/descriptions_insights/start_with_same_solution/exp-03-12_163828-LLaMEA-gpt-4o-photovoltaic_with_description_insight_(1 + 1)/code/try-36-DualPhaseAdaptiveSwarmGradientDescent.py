import numpy as np

class DualPhaseAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.noise_factor = 0.05  # Adjusted noise for better exploration
        self.adaptation_cycle = max(1, dim // 10)  # Adjusted cycle for adaptation
        self.inertia_range = (0.5, 0.9)  # New range for inertia
        self.cognitive_coeff_range = (1.5, 2.2)  # New range for cognitive coeff
        self.social_coeff_range = (1.2, 2.0)  # New range for social coeff

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size
        phase_change = self.budget // 3  # Updated phase change point

        while evaluations < self.budget:
            current_dim = min(self.dim, ((evaluations // self.adaptation_cycle) + 1) * self.adaptation_cycle)
            adaptation_factor = 1 - (evaluations / self.budget) ** 0.8  # Modified adaptation curve
            inertia_weight = self.inertia_range[0] + (self.inertia_range[1] - self.inertia_range[0]) * adaptation_factor
            cognitive_coeff = self.cognitive_coeff_range[1] if evaluations < phase_change else self.cognitive_coeff_range[0]
            social_coeff = self.social_coeff_range[0] if evaluations < phase_change else self.social_coeff_range[1]

            for i in range(self.population_size):
                r1, r2 = np.random.random(current_dim), np.random.random(current_dim)
                learning_rate = 0.35 + 0.1 * adaptation_factor  # Enhanced learning rate dynamics
                self.velocity[i][:current_dim] = (inertia_weight * self.velocity[i][:current_dim] +
                                                  cognitive_coeff * r1 * (personal_best[i][:current_dim] - swarm[i][:current_dim]) +
                                                  social_coeff * r2 * (global_best[:current_dim] - swarm[i][:current_dim]))
                swarm[i][:current_dim] += learning_rate * self.velocity[i][:current_dim]
                swarm[i] = np.clip(swarm[i], lb, ub)

                noise = np.random.normal(0, self.noise_factor * adaptation_factor, current_dim)  # Adapted noise
                swarm[i][:current_dim] += noise
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