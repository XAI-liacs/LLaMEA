import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.evaluations = 0
        
    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
    
    def evaluate_population(self, func, population):
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += len(population)
        return fitness
    
    def opposition_based_learning(self, population):
        opposite_population = self.lower_bound + self.upper_bound - population
        return opposite_population
    
    def mutate(self, population, best_idx):
        indices = np.arange(self.population_size)
        np.random.shuffle(indices)
        
        mutants = []
        for i in range(self.population_size):
            idxs = indices[indices != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = a + self.mutation_factor * (b - c)
            mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
            mutants.append(mutant)
        return np.array(mutants)
    
    def crossover(self, population, mutants):
        cross_points = np.random.rand(self.population_size, self.dim) < self.crossover_rate
        cross_points[np.arange(self.population_size), np.random.randint(0, self.dim, self.population_size)] = True
        trial_population = np.where(cross_points, mutants, population)
        return trial_population
    
    def select(self, population, trial_population, fitness, trial_fitness):
        improved = trial_fitness < fitness
        population[improved] = trial_population[improved]
        fitness[improved] = trial_fitness[improved]
        return population, fitness
    
    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(func, population)
        
        while self.evaluations < self.budget:
            opposite_population = self.opposition_based_learning(population)
            opposite_fitness = self.evaluate_population(func, opposite_population)
            for i in range(self.population_size):
                if opposite_fitness[i] < fitness[i]:
                    population[i] = opposite_population[i]
                    fitness[i] = opposite_fitness[i]

            best_idx = np.argmin(fitness)
            mutants = self.mutate(population, best_idx)
            trial_population = self.crossover(population, mutants)
            trial_fitness = self.evaluate_population(func, trial_population)
            population, fitness = self.select(population, trial_population, fitness, trial_fitness)
        
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]