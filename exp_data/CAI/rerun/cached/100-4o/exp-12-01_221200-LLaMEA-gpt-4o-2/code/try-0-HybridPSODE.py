import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5 # Cognitive (personal) weight
        self.c2 = 1.5 # Social (global) weight
        self.F = 0.8  # Differential evolution mutation factor
        self.CR = 0.9 # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_scores = np.full(self.population_size, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf

    def __call__(self, func):
        eval_count = 0

        while eval_count < self.budget:
            # PSO Update
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest_positions[i] - self.population[i]) +
                                      self.c2 * r2 * (self.gbest_position - self.population[i]))
                self.population[i] += self.velocities[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)
            
            # Evaluate fitness
            for i in range(self.population_size):
                fitness = func(self.population[i])
                eval_count += 1
                if fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = fitness
                    self.pbest_positions[i] = self.population[i]
                if fitness < self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = self.population[i]
                    
                if eval_count >= self.budget:
                    return self.gbest_position

            # DE Mutation and Crossover
            for i in range(self.population_size):
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.pbest_positions[a] + self.F * (self.pbest_positions[b] - self.pbest_positions[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                trial_vector = np.copy(self.population[i])
                crossover = np.random.rand(self.dim) < self.CR
                trial_vector[crossover] = mutant_vector[crossover]
                
                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                eval_count += 1
                if trial_fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = trial_fitness
                    self.pbest_positions[i] = trial_vector
                if trial_fitness < self.gbest_score:
                    self.gbest_score = trial_fitness
                    self.gbest_position = trial_vector
                
                if eval_count >= self.budget:
                    return self.gbest_position
        return self.gbest_position