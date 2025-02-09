import numpy as np

class AdaptiveGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_rate = 0.1
        self.elite_fraction = 0.1
        self.best_solution = None
        self.best_obj = float('inf')
    
    def initialize_population(self, bounds):
        self.lb, self.ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.array([float('inf')] * self.population_size)
    
    def evaluate_population(self, func):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.best_obj:
                self.best_obj = self.fitness[i]
                self.best_solution = self.population[i]
    
    def select_parents(self):
        probabilities = 1 / (self.fitness + 1e-9)
        probabilities /= probabilities.sum()
        parents_indices = np.random.choice(self.population_size, size=self.population_size, p=probabilities)
        return self.population[parents_indices]
    
    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        offspring = alpha * parent1 + (1 - alpha) * parent2
        return offspring
    
    def mutate(self, individual):
        mutation_prob = np.random.rand(self.dim) < self.mutation_rate
        mutation_values = np.random.uniform(self.lb, self.ub, self.dim)
        individual[mutation_prob] = mutation_values[mutation_prob]
        return individual
    
    def create_next_generation(self, parents):
        next_generation = []
        num_elites = int(self.population_size * self.elite_fraction)
        elite_indices = np.argsort(self.fitness)[:num_elites]
        next_generation.extend(self.population[elite_indices])
        
        while len(next_generation) < self.population_size:
            parent1, parent2 = parents[np.random.choice(self.population_size, 2, replace=False)]
            offspring = self.crossover(parent1, parent2)
            offspring = self.mutate(offspring)
            next_generation.append(offspring)
        
        return np.array(next_generation)
    
    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size
            
            parents = self.select_parents()
            self.population = self.create_next_generation(parents)
            
            # Adapt mutation rate based on budget usage
            self.mutation_rate = 0.1 * (1 - (evaluations / self.budget))
        
        return self.best_solution