import numpy as np

class AdaptiveSwarmGradientDescent:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 10 + 2 * int(np.sqrt(dim))
        self.velocity = np.zeros((self.initial_population_size, dim))
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm = np.random.uniform(lb, ub, (self.initial_population_size, self.dim))
        personal_best = swarm.copy()
        personal_best_value = np.array([func(x) for x in swarm])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.initial_population_size
        while evaluations < self.budget:
            adaptive_factor = 1 - evaluations / self.budget
            inertia_weight = 0.7 + 0.3 * np.sin(adaptive_factor * np.pi)  # Adjusted strategy
            cognitive_coeff = 1.1 + 0.5 * adaptive_factor  # Adjusted strategy
            social_coeff = 1.8 + 0.2 * np.cos(adaptive_factor * np.pi)  # Adjusted strategy
            learning_rate = 0.2 + 0.6 * adaptive_factor  # Adjusted strategy
            
            current_population_size = self.initial_population_size  # Fixed population size

            for i in range(current_population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                neighborhood_influence = 0.8 + 0.2 * np.random.rand()

                self.velocity[i] = (inertia_weight * self.velocity[i] +
                                    cognitive_coeff * r1 * (personal_best[i % self.initial_population_size] - swarm[i]) +
                                    social_coeff * r2 * (global_best - swarm[i]) * neighborhood_influence)
                swarm[i] += learning_rate * self.velocity[i]
                swarm[i] = np.clip(swarm[i], lb, ub)

                f_value = func(swarm[i])
                evaluations += 1
                if f_value < personal_best_value[i % self.initial_population_size]:
                    personal_best[i % self.initial_population_size] = swarm[i]
                    personal_best_value[i % self.initial_population_size] = f_value

                if f_value < global_best_value:
                    global_best = swarm[i]
                    global_best_value = f_value

                if np.random.rand() < 0.12:  # Reduced mutation probability
                    mutation_strength = 0.3 * np.random.rand() * (1 - adaptive_factor)  # Adjusted mutation strength
                    mutation = np.random.normal(0, mutation_strength, self.dim)
                    swarm[i] = np.clip(swarm[i] + mutation, lb, ub)
                
                if np.random.rand() < 0.05 * (1 - adaptive_factor):
                    swarm[i] = np.random.uniform(lb, ub, self.dim)

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value