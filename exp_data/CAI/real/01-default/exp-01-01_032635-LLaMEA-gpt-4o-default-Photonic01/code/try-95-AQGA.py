import numpy as np

class AQGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = np.inf
        self.bounds = None
    
    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.bounds = (lb, ub)
    
    def evaluate_population(self, func):
        evaluations = 0
        for i in range(self.population_size):
            if evaluations >= self.budget:
                break
            current_fitness = func(self.population[i])
            evaluations += 1
            self.fitness[i] = current_fitness
            if current_fitness < self.best_fitness:
                self.best_fitness = current_fitness
                self.best_individual = self.population[i].copy()
        return evaluations
    
    def quantum_crossover(self):
        parent1, parent2 = self.select_parents()
        beta = np.random.uniform(-0.1, 1.1, self.dim)
        child1 = parent1 * beta + parent2 * (1 - beta)
        child2 = parent2 * beta + parent1 * (1 - beta)
        return child1, child2
    
    def select_parents(self):
        indices = np.random.choice(self.population_size, 2, replace=False, p=self.fitness_probs())
        return self.population[indices[0]], self.population[indices[1]]
    
    def fitness_probs(self):
        inv_fitness = 1.0 / (self.fitness + 1e-10)
        return inv_fitness / inv_fitness.sum()
    
    def mutate(self, individual):
        mutation_strength = np.random.exponential(0.1, self.dim)
        mutation_vector = np.random.normal(0, mutation_strength, self.dim)
        mutated = individual + mutation_vector
        lb, ub = self.bounds
        return np.clip(mutated, lb, ub)

    def adaptive_mutation(self):
        mutation_rate = 0.2 * (1 - (self.best_fitness / (np.mean(self.fitness) + 1e-10)))
        return mutation_rate
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = self.evaluate_population(func)

        while evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size // 2):
                child1, child2 = self.quantum_crossover()
                if np.random.rand() < self.adaptive_mutation():
                    child1 = self.mutate(child1)
                if np.random.rand() < self.adaptive_mutation():
                    child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = np.array(new_population[:self.population_size])
            evaluations += self.evaluate_population(func)
        
        return self.best_individual, self.best_fitness