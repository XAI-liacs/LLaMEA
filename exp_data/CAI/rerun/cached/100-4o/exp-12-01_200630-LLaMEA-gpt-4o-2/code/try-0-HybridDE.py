import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = min(10 * dim, budget // 5)
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.populations = 3
        self.evaluations = 0

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
    
    def differential_mutation(self, population):
        indices = np.random.choice(np.arange(self.population_size), size=3, replace=False)
        a, b, c = population[indices]
        mutant_vector = a + self.mutation_factor * (b - c)
        return np.clip(mutant_vector, self.bounds[0], self.bounds[1])
    
    def crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.crossover_rate
        trial_vector = np.where(crossover, mutant, target)
        return trial_vector
    
    def select(self, target, trial, func):
        target_fitness = func(target)
        trial_fitness = func(trial)
        self.evaluations += 2
        return (trial, trial_fitness) if trial_fitness < target_fitness else (target, target_fitness)

    def __call__(self, func):
        populations = [self.initialize_population() for _ in range(self.populations)]
        best_solution = None
        best_fitness = float('inf')
        
        while self.evaluations < self.budget:
            for population in populations:
                for i in range(self.population_size):
                    target = population[i]
                    mutant = self.differential_mutation(population)
                    trial = self.crossover(target, mutant)
                    best_candidate, candidate_fitness = self.select(target, trial, func)
                    population[i] = best_candidate
                    
                    if candidate_fitness < best_fitness:
                        best_solution, best_fitness = best_candidate, candidate_fitness

                    if self.evaluations >= self.budget:
                        break
                if self.evaluations >= self.budget:
                    break

        return best_solution