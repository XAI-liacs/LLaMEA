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

        while self.budget > 0:
            ranks = np.argsort(fitness)
            scaled_fitness = 1.0 / (1.0 + ranks)
            # Tournament selection
            parents = np.array([self.tournament_selection(population, fitness, 3) for _ in range(population_size)])
            
            offspring = []
            for i in range(population_size):
                parent1, parent2 = parents[i], parents[(i + 1) % population_size]  # Removed third parent
                alpha = np.random.uniform(0, 1, self.dim)
                child = alpha * parent1 + (1 - alpha) * parent2  # Two-point crossover

                # Adaptive mutation based on convergence history
                history_factor = fitness[best_idx] / (fitness.mean() + 1e-9)
                mutation_strength = 0.1 * (func.bounds.ub - func.bounds.lb) * history_factor
                mutation = np.random.normal(0, mutation_strength, self.dim)
                if np.random.rand() < 0.3 + 0.5 * history_factor:  # Updated adaptive mutation probability
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
        
        return best_solution

    def tournament_selection(self, population, fitness, tournament_size):
        """Tournament selection process to enhance diversity."""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        selected_idx = indices[np.argmin(fitness[indices])]
        return population[selected_idx]