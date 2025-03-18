import numpy as np

class DualPhaseAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 12 + 2 * int(np.sqrt(dim))  # Slightly increased population size
        self.velocity = np.zeros((self.population_size, dim))
        self.noise_factor = 0.03  # Reduced noise factor for precision
        self.layer_increment = max(1, dim // 4)  # Modified increment for better exploration

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size
        phase_change1 = self.budget // 4  # Introduce additional phase change
        phase_change2 = self.budget // 2  # Adjusted second phase transition

        while evaluations < self.budget:
            current_dim = min(self.dim, ((evaluations // self.layer_increment) + 1) * self.layer_increment)
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.6 - 0.1 * adaptive_factor  # Lower inertia for quicker convergence
            cognitive_coeff = 1.8 if evaluations < phase_change1 else 1.35  # Tweaked cognitive coefficient
            social_coeff = 2.0 if evaluations < phase_change2 else 2.1  # Adjusted social coefficient

            for i in range(self.population_size):
                r1, r2 = np.random.random(current_dim), np.random.random(current_dim)
                learning_rate = 0.5 + 0.03 * adaptive_factor  # Adjusted learning rate
                self.velocity[i][:current_dim] = (inertia_weight * self.velocity[i][:current_dim] +
                                                  cognitive_coeff * r1 * (personal_best[i][:current_dim] - swarm[i][:current_dim]) +
                                                  social_coeff * r2 * (global_best[:current_dim] - swarm[i][:current_dim]))
                swarm[i][:current_dim] += learning_rate * self.velocity[i][:current_dim]
                swarm[i] = np.clip(swarm[i], lb, ub)

                noise = np.random.uniform(-self.noise_factor * (adaptive_factor ** 1.5), 
                                           self.noise_factor * (adaptive_factor ** 1.5), current_dim)  # Adaptive noise decay
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