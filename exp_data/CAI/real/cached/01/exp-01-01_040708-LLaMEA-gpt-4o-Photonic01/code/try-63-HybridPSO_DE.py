import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40  # Number of particles
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.f = 0.5  # DE mutation factor
        self.cr = 0.9  # DE crossover probability
        self.best_global_position = None
        self.best_global_fitness = float('inf')
    
    def __call__(self, func):
        np.random.seed(42)
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        swarm_velocity = np.zeros((self.population_size, self.dim))
        personal_best_position = np.copy(swarm_position)
        personal_best_fitness = np.full(self.population_size, float('inf'))

        num_evaluations = 0
        while num_evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(swarm_position[i])
                num_evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_position[i] = swarm_position[i]

                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_position = swarm_position[i]

            adapt_factor = 1 - num_evaluations / self.budget  # New adaptive factor
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.c1 = 1.5 + 0.7 * adapt_factor  # Modify cognitive component
                self.c2 = 1.5 - 0.7 * adapt_factor  # Modify social component
                self.w = 0.4 + 0.65 * (1 - adapt_factor**2)  # Nonlinear inertia weight
                swarm_velocity[i] = (
                    self.w * swarm_velocity[i] +
                    self.c1 * r1 * (personal_best_position[i] - swarm_position[i]) +
                    self.c2 * r2 * (self.best_global_position - swarm_position[i])
                )
                swarm_position[i] += swarm_velocity[i]
                swarm_position[i] = np.clip(swarm_position[i], lb, ub)

                # Differential Evolution mutation and crossover
                if num_evaluations < self.budget:
                    dynamic_f = 0.5 + (0.5 * np.sin(num_evaluations / self.budget * np.pi))  # Adaptive mutation factor
                    dynamic_cr = 0.8 + 0.2 * (1 - adapt_factor)  # Dynamic crossover probability
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant_vector = swarm_position[a] + dynamic_f * (swarm_position[b] - swarm_position[c])
                    trial_vector = np.copy(swarm_position[i])
                    
                    for j in range(self.dim):
                        if np.random.rand() < dynamic_cr:  # Use dynamic_cr instead of self.cr
                            trial_vector[j] = mutant_vector[j]
                    
                    trial_vector = np.clip(trial_vector, lb, ub)
                    trial_fitness = func(trial_vector)
                    num_evaluations += 1

                    if trial_fitness < fitness:
                        swarm_position[i] = trial_vector
                        personal_best_fitness[i] = trial_fitness
                        personal_best_position[i] = trial_vector
                        if trial_fitness < self.best_global_fitness:
                            self.best_global_fitness = trial_fitness
                            self.best_global_position = trial_vector

        return self.best_global_position, self.best_global_fitness