import numpy as np

class AQEHA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.5
        self.elite_fraction = 0.15  # Adjusted to increase elite size
        self.dynamic_population_size = True  # New line

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
            if self.dynamic_population_size:  # New line
                self.population_size = max(10, int(self.population_size * 0.9))  # New line
            for _ in range(self.population_size - elite_size):
                parent1, parent2 = self.select_parents(population, fitness)
                offspring = self.quantum_crossover(parent1, parent2, lb, ub)
                offspring = self.diversity_mutation(offspring, lb, ub)  # Modified line
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
        idx1, idx2 = np.random.choice(len(population), size=2, replace=False)
        return population[idx1], population[idx2]
    
    def quantum_crossover(self, parent1, parent2, lb, ub):
        q1 = self.alpha * parent1 + (1 - self.alpha) * parent2
        q2 = self.alpha * parent2 + (1 - self.alpha) * parent1
        offspring = q1 if np.random.rand() < 0.5 else q2
        offspring = np.clip(offspring, lb, ub)
        return offspring
    
    def diversity_mutation(self, individual, lb, ub):  # Modified function
        mutation_strength = 0.15 * (ub - lb) * np.random.rand(self.dim)  # Adjusted mutation strength
        mutated = individual + np.random.normal(0, mutation_strength, self.dim)
        return np.clip(mutated, lb, ub)

    def selection(self, population, fitness, new_population, new_fitness):
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.concatenate((fitness, new_fitness))
        sorted_indices = np.argsort(combined_fitness)
        return combined_population[sorted_indices][:self.population_size], combined_fitness[sorted_indices][:self.population_size]