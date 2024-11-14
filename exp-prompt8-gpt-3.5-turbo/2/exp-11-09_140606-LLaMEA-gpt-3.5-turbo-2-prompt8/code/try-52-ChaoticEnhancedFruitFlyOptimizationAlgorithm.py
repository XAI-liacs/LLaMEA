import numpy as np

class ChaoticEnhancedFruitFlyOptimizationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.step_size = 1.0

    def chaotic_map(self, x):
        return 3.9 * x * (1 - x)

    def __call__(self, func):
        population_size = 10
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness_values = np.array([func(individual) for individual in population])
        
        for _ in range(self.budget - population_size):
            mean_individual = np.mean(population, axis=0)
            chaotic_value = self.chaotic_map(np.random.rand())
            new_individual = mean_individual + chaotic_value * (np.random.uniform(-1, 1, self.dim) * self.step_size)
            new_fitness = func(new_individual)
            
            if new_fitness < np.max(fitness_values):
                max_idx = np.argmax(fitness_values)
                population[max_idx] = new_individual
                fitness_values[max_idx] = new_fitness
                self.step_size *= 1.1  # Increase step size for better exploration
            else:
                self.step_size *= 0.9  # Decrease step size for better exploitation
                
            if np.random.rand() < 0.1:  # Introduce dynamic population adaptation
                if new_fitness < np.min(fitness_values):
                    population = np.vstack((population, new_individual))
                    fitness_values = np.append(fitness_values, new_fitness)
                    population_size += 1
                elif new_fitness < np.max(fitness_values):
                    replace_idx = np.argmax(fitness_values)
                    population[replace_idx] = new_individual
                    fitness_values[replace_idx] = new_fitness

        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]
        best_fitness = fitness_values[best_idx]
        
        return best_solution, best_fitness