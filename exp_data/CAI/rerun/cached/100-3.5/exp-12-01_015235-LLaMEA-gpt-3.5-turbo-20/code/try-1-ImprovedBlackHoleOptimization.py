import numpy as np

class ImprovedBlackHoleOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_init = 0.9
        self.w_final = 0.4
        
    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        
        def evaluate_population(population):
            return np.array([func(individual) for individual in population])
        
        def update_position(particle, global_best, w):
            r = np.random.uniform(0, 1, self.dim)
            new_particle = particle + w * r * (global_best - particle)
            new_particle = np.clip(new_particle, self.lower_bound, self.upper_bound)
            return new_particle
        
        population = initialize_population()
        fitness_values = evaluate_population(population)
        global_best_index = np.argmin(fitness_values)
        global_best = population[global_best_index]
        
        w = self.w_init
        w_decay = (self.w_init - self.w_final) / self.budget
        
        for _ in range(self.budget):
            for i in range(self.budget):
                new_position = update_position(population[i], global_best, w)
                new_fitness = func(new_position)
                if new_fitness < fitness_values[i]:
                    fitness_values[i] = new_fitness
                    population[i] = new_position
                    if new_fitness < fitness_values[global_best_index]:
                        global_best_index = i
                        global_best = new_position
            w -= w_decay
        
        return global_best