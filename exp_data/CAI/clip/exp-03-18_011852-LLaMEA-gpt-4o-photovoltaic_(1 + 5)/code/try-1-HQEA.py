import numpy as np

class HQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.5
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = self.population_size
        
        while evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents(population, fitness)
                offspring = self.quantum_crossover(parent1, parent2, lb, ub)
                new_population.append(offspring)
            
            new_population = np.array(new_population)
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size
            
            population, fitness = self.selection(population, fitness, new_population, new_fitness)
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution, best_fitness = population[current_best_idx], fitness[current_best_idx]
        
        return best_solution, best_fitness
    
    def select_parents(self, population, fitness):
        idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
        return population[idx1], population[idx2]
    
    def quantum_crossover(self, parent1, parent2, lb, ub):
        beta = np.random.rand()  # Line modified to make crossover adaptive
        q1 = beta * parent1 + (1 - beta) * parent2
        q2 = beta * parent2 + (1 - beta) * parent1
        offspring = q1 if np.random.rand() < 0.5 else q2
        offspring = np.clip(offspring, lb, ub)
        return offspring
    
    def selection(self, population, fitness, new_population, new_fitness):
        elite_idx = np.argmin(new_fitness) # Line modified to include elitism
        if new_fitness[elite_idx] < np.min(fitness):
            best_elite = new_population[elite_idx] # Line modified to include elitism
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.concatenate((fitness, new_fitness))
        sorted_indices = np.argsort(combined_fitness)
        return np.vstack((best_elite, combined_population[sorted_indices][:self.population_size-1])), np.concatenate(([new_fitness[elite_idx]], combined_fitness[sorted_indices][:self.population_size-1])) # Modified to ensure elite is included