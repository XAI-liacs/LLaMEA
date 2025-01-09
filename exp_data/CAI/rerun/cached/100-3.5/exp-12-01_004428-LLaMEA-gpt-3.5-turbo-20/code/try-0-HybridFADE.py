import numpy as np

class HybridFADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.firefly_alpha = 0.5
        self.de_cr = 0.5
        self.de_f = 0.5

    def __call__(self, func):
        def init_population():
            return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        
        def firefly_move(firefly, best_firefly):
            beta = 1 / (1 + self.firefly_alpha * np.linalg.norm(firefly - best_firefly))
            new_pos = firefly + beta * (best_firefly - firefly) + 0.01 * np.random.normal(size=self.dim)
            return np.clip(new_pos, -5.0, 5.0)
        
        def de_mutation(population):
            donor = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                candidates = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                donor[i] = np.clip(population[a] + self.de_f * (population[b] - population[c]), -5.0, 5.0)
            return donor
        
        population = init_population()
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        for _ in range(self.budget - self.population_size):
            donor = de_mutation(population)
            for i in range(self.population_size):
                if func(donor[i]) < fitness[i]:
                    population[i] = donor[i]
                    fitness[i] = func(donor[i])
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            
            for i in range(self.population_size):
                population[i] = firefly_move(population[i], best_solution)
        
        return best_solution