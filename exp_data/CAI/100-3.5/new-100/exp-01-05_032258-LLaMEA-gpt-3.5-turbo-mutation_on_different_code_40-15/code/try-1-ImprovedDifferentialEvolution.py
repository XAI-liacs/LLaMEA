import numpy as np

class ImprovedDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population_size = 10
        scaling_factor = 0.8
        crossover_prob = 0.9
        scaling_factor_lower = 0.5
        scaling_factor_upper = 1.0
        
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for i in range(self.budget):
            for j in range(population_size):
                candidate = population[j]
                indices = [idx for idx in range(population_size) if idx != j]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                scaling_factor = np.random.uniform(scaling_factor_lower, scaling_factor_upper)
                
                mutated_vector = candidate + scaling_factor * (a - b)
                crossover_mask = np.random.rand(self.dim) < crossover_prob
                trial_vector = np.where(crossover_mask, mutated_vector, candidate)
                
                trial_fitness = func(trial_vector)
                if trial_fitness <= fitness[j]:
                    population[j] = trial_vector
                    fitness[j] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector
                else:
                    if np.random.rand() < 0.1:  # Introduce elitism
                        population[j] = trial_vector
                        fitness[j] = trial_fitness
            
        return self.f_opt, self.x_opt