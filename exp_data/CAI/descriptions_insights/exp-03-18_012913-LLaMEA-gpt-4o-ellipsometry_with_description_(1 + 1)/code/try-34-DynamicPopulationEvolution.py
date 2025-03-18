# Code for DynamicPopulationEvolution class
import numpy as np
import random

class DynamicPopulationEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define population size
        population_size = min(20, self.budget // 2)
        # Initialize population within bounds
        population = np.random.uniform(
            low=func.bounds.lb, 
            high=func.bounds.ub, 
            size=(population_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in population])
        self.budget -= population_size
        best_idx = np.argmin(fitness)  # Track best solution
        best_solution = population[best_idx]

        # Evolution loop
        while self.budget > 0:
            # Rank-based fitness scaling
            ranks = np.argsort(fitness)
            scaled_fitness = 1.0 / (1.0 + ranks)
            parents_idx = np.random.choice(
                population_size, 
                size=population_size,
                p=scaled_fitness / scaled_fitness.sum()
            )
            parents = population[parents_idx]
            
            # Generate offspring through crossover and mutation
            offspring = []
            adaptive_mutation_prob = 0.1 + 0.4 * (fitness[best_idx] / fitness.mean())  # Adaptive mutation probability
            for i in range(population_size):
                # Introduced ensemble crossover
                parent1, parent2, parent3 = parents[i], parents[(i + 1) % population_size], parents[(i + 2) % population_size]  # Added a third parent

                # Crossover (blend crossover)
                alpha = np.random.uniform(0, 1, self.dim)
                child = alpha * parent1 + (1 - alpha) * (0.5 * (parent2 + parent3))  # Modified crossover

                # Mutation (adaptive Gaussian mutation)
                population_diversity = np.std(population, axis=0)  # Calculate population diversity
                mutation_strength = 0.1 * (func.bounds.ub - func.bounds.lb) * (1 + population_diversity.mean())  # Updated mutation strength
                mutation = np.random.normal(0, mutation_strength, self.dim)
                if np.random.rand() < adaptive_mutation_prob:
                    child += mutation
                
                # Ensure child is within bounds
                child = np.clip(child, func.bounds.lb, func.bounds.ub)
                offspring.append(child)
            
            offspring = np.array(offspring)
            offspring_fitness = np.array([func(ind) for ind in offspring])
            self.budget -= population_size
            
            # Combine and select the next generation
            combined_population = np.vstack((population, offspring, [best_solution]))  # Preserve best solution
            combined_fitness = np.hstack((fitness, offspring_fitness, [fitness[best_idx]]))
            best_individuals_idx = np.argsort(combined_fitness)[:population_size]
            population = combined_population[best_individuals_idx]
            fitness = combined_fitness[best_individuals_idx]
            
            # Update best solution
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
        
        # Return the best found solution
        return best_solution