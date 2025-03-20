import numpy as np

class AQDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.5
        self.elite_fraction = 0.1
        self.mutation_factor = 0.8  # New line
        self.crossover_rate = 0.7  # New line

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
            elite_size = int(self.elite_fraction * self.population_size)
            elites = population[np.argsort(fitness)[:elite_size]]
            for i in range(self.population_size - elite_size):
                parent1, parent2, parent3 = self.select_parents(population, fitness)  # Modified line
                mutant = self.differential_mutation(parent1, parent2, parent3, lb, ub)  # New line
                offspring = self.quantum_crossover(mutant, population[i], lb, ub)  # Modified line
                new_population.append(offspring)
            
            new_population.extend(elites)
            new_population = np.array(new_population)
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size
            
            population, fitness = self.selection(population, fitness, new_population, new_fitness)
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_solution, best_fitness = population[current_best_idx], fitness[current_best_idx]
        
        return best_solution, best_fitness
    
    def select_parents(self, population, fitness):
        indices = np.random.choice(len(population), size=3, replace=False)  # Modified line
        return population[indices[0]], population[indices[1]], population[indices[2]]  # Modified line
    
    def quantum_crossover(self, parent1, parent2, lb, ub):
        if np.random.rand() < self.crossover_rate:  # New line
            offspring = self.alpha * parent1 + (1 - self.alpha) * parent2  # Modified line
        else:  # New line
            offspring = parent2  # New line
        return np.clip(offspring, lb, ub)

    def differential_mutation(self, parent1, parent2, parent3, lb, ub):  # New function
        mutant = parent1 + self.mutation_factor * (parent2 - parent3)
        return np.clip(mutant, lb, ub)

    def selection(self, population, fitness, new_population, new_fitness):
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.concatenate((fitness, new_fitness))
        sorted_indices = np.argsort(combined_fitness)
        return combined_population[sorted_indices][:self.population_size], combined_fitness[sorted_indices][:self.population_size]