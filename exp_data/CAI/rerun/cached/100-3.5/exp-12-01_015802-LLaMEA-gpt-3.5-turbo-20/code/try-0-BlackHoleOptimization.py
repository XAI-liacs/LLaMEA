import numpy as np

class BlackHoleOptimization:
    def __init__(self, budget, dim, num_holes=10, num_iterations=100):
        self.budget = budget
        self.dim = dim
        self.num_holes = num_holes
        self.num_iterations = num_iterations

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, size=(self.num_holes, self.dim))

        def calculate_fitness(population):
            return np.array([func(individual) for individual in population])

        def update_position(population, fitness):
            best_idx = np.argmax(fitness)
            centroid = np.mean(population, axis=0)
            new_population = population.copy()
            for i in range(self.num_holes):
                if i != best_idx:
                    direction = population[i] - population[best_idx]
                    distance = np.linalg.norm(population[i] - population[best_idx])
                    new_population[i] = population[i] + np.random.uniform() * 2 * direction / distance + centroid - population[i]
            return new_population

        population = initialize_population()
        fitness = calculate_fitness(population)
        
        for _ in range(self.num_iterations):
            new_population = update_position(population, fitness)
            new_fitness = calculate_fitness(new_population)
            if np.max(new_fitness) > np.max(fitness):
                population = new_population
                fitness = new_fitness

        best_idx = np.argmax(fitness)
        best_solution = population[best_idx]
        
        return best_solution