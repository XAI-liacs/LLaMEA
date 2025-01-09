import numpy as np

class AdaptiveMPGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20
        self.num_parents = 10
        self.sigma = 0.1
        self.tau = 0.5
        self.elite_rate = 0.1
        self.parents = None
        self.initialized = False

    def initialize_population(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                            (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.best_individual = None
        self.best_fitness = np.inf
        self.initialized = True

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])

    def select_parents(self):
        indices = np.argsort(self.fitness)[:self.num_parents]
        self.parents = self.population[indices]

    def crossover(self):
        offspring = []
        num_offspring = self.pop_size - len(self.parents)
        for _ in range(num_offspring):
            p1, p2 = self.parents[np.random.choice(self.num_parents, 2, replace=False)]
            alpha = np.random.uniform(0.0, 1.0, self.dim)
            child = alpha * p1 + (1 - alpha) * p2
            offspring.append(child)
        return np.array(offspring)

    def mutate(self, offspring):
        mutation_rate = 1 / self.dim  # Adaptive mutation rate
        for i in range(len(offspring)):
            if np.random.rand() < self.tau:
                mutation = np.random.normal(0, self.sigma * mutation_rate, self.dim)
                offspring[i] += mutation
                offspring[i] = np.clip(offspring[i], self.lower_bound, self.upper_bound)
        return offspring

    def elitism(self):
        num_elites = int(self.elite_rate * self.pop_size)
        elite_indices = np.argsort(self.fitness)[:num_elites]
        return self.population[elite_indices]

    def __call__(self, func):
        if not self.initialized:
            self.initialize_population()

        func_eval_count = 0
        while func_eval_count < self.budget:
            self.evaluate_population(func)
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_individual = self.population[best_idx].copy()
            
            self.select_parents()
            offspring = self.crossover()
            offspring = self.mutate(offspring)
            elites = self.elitism()
            self.population = np.vstack((offspring, elites))
            self.pop_size = len(self.population)  # Update population size dynamically
            self.fitness = np.full(self.pop_size, np.inf)
            
            func_eval_count += self.pop_size
        
        return self.best_individual, self.best_fitness