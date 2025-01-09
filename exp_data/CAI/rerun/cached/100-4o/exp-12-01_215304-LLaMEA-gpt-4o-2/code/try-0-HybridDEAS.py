import numpy as np

class HybridDEAS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + dim  # Adjust size based on dimension
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.scale_factor = 0.8
        self.crossover_rate = 0.9
        self.min_budget = budget
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)

    def __call__(self, func):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            
        used_budget = self.population_size
        
        while used_budget < self.min_budget:
            # Differential Evolution (DE) Phase
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.population[a] + self.scale_factor * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                
                crossover_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, self.population[i])
                trial_fitness = func(crossover_vector)
                used_budget += 1
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = crossover_vector
                    self.fitness[i] = trial_fitness
                    
            # Adaptive Swarm Intelligence (ASI) Phase
            global_best_idx = np.argmin(self.fitness)
            for i in range(self.population_size):
                if np.random.rand() < 0.5:
                    learning_factor = 0.5 + np.random.rand() / 2.0
                    attraction_vector = self.population[global_best_idx] - self.population[i]
                    step_vector = learning_factor * attraction_vector + np.random.randn(self.dim)
                    self.population[i] = np.clip(self.population[i] + step_vector, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(self.population[i])
                    used_budget += 1
                    
                    if candidate_fitness < self.fitness[i]:
                        self.fitness[i] = candidate_fitness

        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]