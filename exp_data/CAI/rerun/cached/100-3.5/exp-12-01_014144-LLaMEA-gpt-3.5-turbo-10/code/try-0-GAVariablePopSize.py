import numpy as np

class GAVariablePopSize:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.max_pop_size = 50
        self.min_pop_size = 5
        self.pc = 0.8
        self.pm = 1/dim

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))

        def mutate(child):
            mask = np.random.rand(self.dim) < self.pm
            child[mask] = np.random.uniform(-5.0, 5.0, size=np.sum(mask))
            return child

        def crossover(parent1, parent2):
            if np.random.rand() < self.pc:
                idx = np.random.randint(self.dim)
                child = np.concatenate((parent1[:idx], parent2[idx:]))
                return child
            else:
                return parent1

        population = initialize_population()
        fitness = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget):
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            new_population = []
            for i in range(self.pop_size):
                parent1, parent2 = population[np.random.choice(self.pop_size, size=2, replace=False)]
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            
            new_population = np.array(new_population)
            new_fitness = np.array([func(individual) for individual in new_population])
            population = np.vstack((population, new_population))
            fitness = np.concatenate((fitness, new_fitness))
            
            diversity = np.mean(np.std(population, axis=0))
            if diversity < 1e-6:
                break

            if len(population) > self.max_pop_size:
                idx = np.argsort(fitness)[:self.min_pop_size]
                population = population[idx]
                fitness = fitness[idx]
                
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        return best_solution