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
            inertia_weight = 0.5 + 0.3 * np.sin(np.pi * adaptive_factor**2)  # Modified coefficient
            cognitive_coeff = 1.6 * adaptive_factor  # Modified coefficient
            social_coeff = 1.2 + 0.2 * np.cos(np.pi * adaptive_factor)  # Modified coefficient
            learning_rate = 0.1 + 0.9 * adaptive_factor  # Modified coefficient
            
            current_population_size = int(self.initial_population_size * (1 + 0.6 * adaptive_factor))  # Modified population scaling
            self.velocity = np.resize(self.velocity, (current_population_size, self.dim))
            swarm = np.resize(swarm, (current_population_size, self.dim))
            
            for i in range(current_population_size):
                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                neighborhood_influence = 0.5 + 0.5 * np.random.rand()  # New term for diversity
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

                if np.random.rand() < 0.15:
                    min_pbv = np.min(personal_best_value)
                    max_pbv = np.max(personal_best_value)
                    mutation_strength = (f_value - min_pbv) / (max_pbv - min_pbv + 1e-10)
                    mutation_rate = 0.2 * (1 - adaptive_factor) * mutation_strength  # Modified mutation rate
                    mutation = np.random.normal(0, mutation_rate, self.dim)
                    swarm[i] = np.clip(swarm[i] + mutation, lb, ub)
                
                if np.random.rand() < 0.02 * (1 - adaptive_factor):  # Modified reinitialization probability
                    swarm[i] = np.random.uniform(lb, ub, self.dim)

                if evaluations >= self.budget:
                    break

        return global_best, global_best_value