import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.swarm = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.local_best_pos = np.copy(self.swarm)
        self.global_best_pos = None
        self.local_best_val = np.full(self.population_size, np.inf)
        self.global_best_val = np.inf
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
    
    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            # Evaluate fitness
            for i in range(self.population_size):
                fitness = func(self.swarm[i])
                evaluations += 1
                if fitness < self.local_best_val[i]:
                    self.local_best_val[i] = fitness
                    self.local_best_pos[i] = self.swarm[i].copy()
                if fitness < self.global_best_val:
                    self.global_best_val = fitness
                    self.global_best_pos = self.swarm[i].copy()
                    
                if evaluations >= self.budget:
                    break
            
            # Update velocities and positions (PSO)
            inertia_weight = 0.7
            cognitive_const = 1.5
            social_const = 1.5
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.velocities = (inertia_weight * self.velocities +
                               cognitive_const * r1 * (self.local_best_pos - self.swarm) +
                               social_const * r2 * (self.global_best_pos - self.swarm))
            self.swarm += self.velocities
            self.swarm = np.clip(self.swarm, self.lower_bound, self.upper_bound)
            
            # Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = self.swarm[indices[0]], self.swarm[indices[1]], self.swarm[indices[2]]
                mutant_vector = x1 + self.mutation_factor * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                trial_vector = np.copy(self.swarm[i])
                crossover_indices = np.random.rand(self.dim) < self.crossover_prob
                trial_vector[crossover_indices] = mutant_vector[crossover_indices]
                
                # Selection
                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < self.local_best_val[i]:
                    self.swarm[i] = trial_vector
                    self.local_best_val[i] = trial_fitness
                    if trial_fitness < self.global_best_val:
                        self.global_best_val = trial_fitness
                        self.global_best_pos = trial_vector

                if evaluations >= self.budget:
                    break
        return self.global_best_pos, self.global_best_val