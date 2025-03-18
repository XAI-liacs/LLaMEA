import numpy as np

class EnhancedAdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.population_size, dim))
        self.mutation_rate = 0.2  # Changed line 1
        self.layer_increase_step = np.ceil(dim / 6).astype(int)  # Changed line 2
        self.initial_dim = 6

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        current_dim = self.initial_dim  # Changed line 3
        swarm = np.random.uniform(lb, ub, (self.population_size, current_dim))  # Changed line 4
        personal_best = swarm.copy()
        personal_best_value = np.array([func(np.pad(x, (0, self.dim-current_dim), 
                                        'constant')) for x in swarm])  # Changed line 5
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            diversity_factor = np.std(swarm, axis=0).mean() / (ub - lb).mean()
            inertia_weight = (0.7 + 0.3 * np.cos(adaptive_factor * np.pi)) * (1 + diversity_factor)
            cognitive_coeff = 1.5 * (1 + 0.5 * adaptive_factor)
            social_coeff = 1.5 * (1 - 0.5 * adaptive_factor)

            if evaluations / self.budget > 0.5 and current_dim < self.dim:  # Changed line 6
                current_dim = min(current_dim + self.layer_increase_step, self.dim)  # Changed line 7
                swarm = np.pad(swarm, ((0, 0), (0, current_dim - swarm.shape[1])), 
                               'constant', constant_values=0)  # Changed line 8

            for i in range(self.population_size):
                r1, r2 = np.random.random(current_dim), np.random.random(current_dim)  # Changed line 9
                role_based_factor = np.sin(np.pi * i / self.population_size)  # Changed line 10
                adaptive_scaling = (0.85 + 0.15 * np.sin(evaluations / self.budget * np.pi))
                self.velocity[i] = (adaptive_scaling * inertia_weight * self.velocity[i][:current_dim] +  # Changed line 11
                                    cognitive_coeff * r1 * (personal_best[i][:current_dim] - swarm[i]) +  # Changed line 12
                                    social_coeff * r2 * (global_best[:current_dim] - swarm[i]) +  # Changed line 13
                                    role_based_factor * np.random.normal(0, 0.1, current_dim))  # Changed line 14
                swarm[i] += self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                # Introduce mutation for diversity
                adjusted_mutation_rate = self.mutation_rate * (1 + diversity_factor)
                if np.random.rand() < adjusted_mutation_rate:
                    non_uniform_scale = np.linalg.norm(global_best[:current_dim] - swarm[i]) / np.sqrt(current_dim)  # Changed line 15
                    mutation_scale = (np.random.normal(0, 0.25 * adaptive_scaling * non_uniform_scale * 
                                    (1 - adaptive_factor), current_dim) * (diversity_factor + 0.1))  # Changed line 16
                    swarm[i] += mutation_scale
                    swarm[i] = np.clip(swarm[i], lb, ub)

                # Evaluate and update personal best
                f_value = func(np.pad(swarm[i], (0, self.dim-current_dim), 'constant'))  # Changed line 17
                evaluations += 1
                if f_value < personal_best_value[i]:
                    personal_best[i] = swarm[i]
                    personal_best_value[i] = f_value

                # Update global best
                if f_value < global_best_value:
                    global_best = np.pad(swarm[i], (0, self.dim-current_dim), 'constant')  # Changed line 18
                    global_best_value = f_value

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value