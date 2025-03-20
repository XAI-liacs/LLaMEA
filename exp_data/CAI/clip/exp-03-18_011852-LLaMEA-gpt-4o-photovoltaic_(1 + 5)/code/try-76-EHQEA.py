import numpy as np

class EHQEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.5
        self.elite_fraction = 0.1

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
            for _ in range(self.population_size - elite_size):
                parent1, parent2 = self.select_parents(population, fitness)
                offspring = self.enhanced_quantum_crossover(parent1, parent2, lb, ub)
                offspring = self.diverse_adaptive_mutation(offspring, lb, ub, evaluations / self.budget)
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
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        parent1_idx = tournament_indices[np.argmin(tournament_fitness)]
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = fitness[tournament_indices]
        parent2_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[parent1_idx], population[parent2_idx]
    
    def enhanced_quantum_crossover(self, parent1, parent2, lb, ub):
        alpha_var = np.random.uniform(0.4, 0.6)
        q1 = alpha_var * parent1 + (1 - alpha_var) * parent2
        q2 = alpha_var * parent2 + (1 - alpha_var) * parent1
        offspring = q1 if np.random.rand() < 0.5 else q2
        offspring = np.clip(offspring, lb, ub)
        return offspring
    
    def diverse_adaptive_mutation(self, individual, lb, ub, progress):
        mutation_strength = 0.1 * (ub - lb) * np.random.random_sample(self.dim) * (1 - progress ** 2.0)
        if np.random.random() < 0.5:
            mutated = individual + np.random.normal(0, mutation_strength, self.dim)
        else:
            mutated = individual + np.random.uniform(-mutation_strength, mutation_strength, self.dim)
        return np.clip(mutated, lb, ub)

    def selection(self, population, fitness, new_population, new_fitness):
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.concatenate((fitness, new_fitness))
        sorted_indices = np.argsort(combined_fitness)
        return combined_population[sorted_indices][:self.population_size], combined_fitness[sorted_indices][:self.population_size]