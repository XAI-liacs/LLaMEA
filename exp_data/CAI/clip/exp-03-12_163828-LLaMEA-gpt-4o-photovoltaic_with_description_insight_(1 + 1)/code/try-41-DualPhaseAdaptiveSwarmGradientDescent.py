import numpy as np

class DualPhaseAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 3 * int(np.sqrt(dim))  # Increased pop size for diversity
        self.velocity = np.zeros((self.population_size, dim))
        self.noise_factor = 0.015  # Reduced noise factor for finer updates
        self.layer_increment = max(1, dim // 5)  # Smaller increment for gradual adaptation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size
        phase_change = self.budget // 3  # Earlier phase transition for quicker convergence

        while evaluations < self.budget:
            current_dim = min(self.dim, ((evaluations // self.layer_increment) + 1) * self.layer_increment)
            adaptive_factor = np.exp(-evaluations / self.budget)  # Exponential decay for adaptation
            inertia_weight = 0.5 + 0.2 * adaptive_factor  # Refined inertia for dynamism
            cognitive_coeff = 1.8 if evaluations < phase_change else 1.2  # Refined cognitive coefficient
            social_coeff = 1.7 if evaluations < phase_change else 2.0  # Enhanced social coefficient

            for i in range(self.population_size):
                r1, r2 = np.random.random(current_dim), np.random.random(current_dim)
                learning_rate = 0.35 + 0.1 * adaptive_factor  # Higher learning rate for speed
                self.velocity[i][:current_dim] = (inertia_weight * self.velocity[i][:current_dim] +
                                                  cognitive_coeff * r1 * (personal_best[i][:current_dim] - swarm[i][:current_dim]) +
                                                  social_coeff * r2 * (global_best[:current_dim] - swarm[i][:current_dim]))
                swarm[i][:current_dim] += learning_rate * self.velocity[i][:current_dim]
                swarm[i] = np.clip(swarm[i], lb, ub)

                noise = np.random.normal(0, self.noise_factor * (adaptive_factor ** 2), current_dim)  # Adjusted noise scaling
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