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
        
        no_improvement_count = 0  # Counter to track convergence

        while self.budget > 0:
            parents_idx = np.random.choice(
                population_size, 
                size=population_size, 
                p=(1 / (fitness + 1e-9)) / (1 / (fitness + 1e-9)).sum()
            )
            parents = population[parents_idx]
            
            offspring = []
            for i in range(population_size):
                parent1, parent2 = parents[i], parents[(i + 1) % population_size]
                alpha = np.random.uniform(0, 1, self.dim)
                child = alpha * parent1 + (1 - alpha) * parent2
                
                mutation_strength = 0.1 * (func.bounds.ub - func.bounds.lb) * np.std(fitness)
                mutation = np.random.normal(0, mutation_strength, self.dim)
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
            
            new_best_idx = np.argmin(fitness)
            if fitness[new_best_idx] < fitness[best_idx]:
                best_idx = new_best_idx
                best_solution = population[best_idx]
                no_improvement_count = 0  # Reset convergence counter
            else:
                no_improvement_count += 1

            if no_improvement_count > 5:  # Restart if no improvement for 5 iterations
                population = np.random.uniform(
                    low=func.bounds.lb, 
                    high=func.bounds.ub, 
                    size=(population_size, self.dim)
                )
                fitness = np.array([func(ind) for ind in population])
                self.budget -= population_size
                no_improvement_count = 0

        return best_solution