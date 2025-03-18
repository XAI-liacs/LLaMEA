import numpy as np
import random

class DynamicPopulationEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = min(20, self.budget // 2)
        population = np.random.uniform(
            low=func.bounds.lb, 
            high=func.bounds.ub, 
            size=(population_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in population])
        self.budget -= population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        
        # New: Maintain a memory of best solutions
        best_memory = [best_solution]

        while self.budget > 0:
            ranks = np.argsort(fitness)
            scaled_fitness = 1.0 / (1.0 + ranks)
            parents_idx = np.random.choice(
                population_size, 
                size=population_size,
                p=scaled_fitness / scaled_fitness.sum()
            )
            parents = population[parents_idx]
            
            offspring = []
            fitness_variance_ratio = fitness.std() / fitness.mean()
            adaptive_mutation_prob = 0.15 + 0.25 * (fitness[best_idx] / fitness.mean())  # Adjusted mutation probability
            mutation_decay_factor = 1 - (self.budget / (self.budget + random.randint(1, population_size)))
            for i in range(population_size):
                parent1, parent2, parent3 = parents[i], parents[(i + 1) % population_size], parents[(i + 2) % population_size]

                alpha = np.random.uniform(0, 1, self.dim)
                centroid = (parent1 + parent2 + parent3) / 3
                child = alpha * parent1 + (1 - alpha) * centroid

                mutation_strength = mutation_decay_factor * 0.1 * (func.bounds.ub - func.bounds.lb) * (fitness[best_idx] / fitness.min())
                mutation = np.random.normal(0, mutation_strength, self.dim)
                # New: Dual mutation strategy
                if np.random.rand() < adaptive_mutation_prob:
                    mutation = np.where(np.random.rand(self.dim) > 0.5, mutation, -mutation)  # Dual mutation strategy
                    child += mutation
                
                child = np.clip(child, func.bounds.lb, func.bounds.ub)
                offspring.append(child)
            
            offspring = np.array(offspring)
            offspring_fitness = np.array([func(ind) for ind in offspring])
            self.budget -= population_size
            
            combined_population = np.vstack((population, offspring, [best_solution]))
            combined_fitness = np.hstack((fitness, offspring_fitness, [fitness[best_idx]]))
            best_individuals_idx = np.argsort(combined_fitness)[:population_size]
            population = combined_population[best_individuals_idx]
            fitness = combined_fitness[best_individuals_idx]
            
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            best_memory.append(best_solution)  # Update memory
        
        # Return the best found solution from memory
        return best_solution