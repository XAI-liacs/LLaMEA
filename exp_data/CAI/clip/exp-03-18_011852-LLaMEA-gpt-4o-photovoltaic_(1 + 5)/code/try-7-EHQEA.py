import numpy as np

class EHQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.5
        self.elite_fraction = 0.2  # Modified line

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
            elite_size = int(self.dynamic_elite_fraction(evaluations) * self.population_size)  # Modified line
            elites = population[np.argsort(fitness)[:elite_size]]
            for _ in range(self.population_size - elite_size):
                parent1, parent2 = self.tournament_selection(population, fitness)  # Modified line
                offspring = self.enhanced_quantum_crossover(parent1, parent2, lb, ub)  # Modified line
                offspring = self.adaptive_mutation(offspring, lb, ub)
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
    
    def tournament_selection(self, population, fitness):  # Modified line
        indices = np.random.choice(len(population), size=4, replace=False)  # Modified line
        selected = indices[np.argsort(fitness[indices])[:2]]  # Modified line
        return population[selected[0]], population[selected[1]]  # Modified line
    
    def enhanced_quantum_crossover(self, parent1, parent2, lb, ub):  # Modified line
        beta = np.random.rand(self.dim)  # Modified line
        offspring = beta * parent1 + (1 - beta) * parent2  # Modified line
        return np.clip(offspring, lb, ub)  # Modified line
    
    def adaptive_mutation(self, individual, lb, ub):
        mutation_strength = 0.1 * (ub - lb) * np.random.rand(self.dim)
        mutated = individual + np.random.normal(0, mutation_strength, self.dim)
        return np.clip(mutated, lb, ub)

    def dynamic_elite_fraction(self, evaluations):  # New function
        return max(0.1, self.elite_fraction - 0.05 * (evaluations / self.budget))  # Modified line

    def selection(self, population, fitness, new_population, new_fitness):
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.concatenate((fitness, new_fitness))
        sorted_indices = np.argsort(combined_fitness)
        return combined_population[sorted_indices][:self.population_size], combined_fitness[sorted_indices][:self.population_size]