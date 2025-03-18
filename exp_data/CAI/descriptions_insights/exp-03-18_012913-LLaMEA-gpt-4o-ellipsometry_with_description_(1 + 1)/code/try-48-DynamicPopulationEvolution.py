import numpy as np
import random

class DynamicPopulationEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define population size
        population_size = min(20, self.budget // 2)
        # Initialize multiple swarms
        num_swarms = 3  # New: Utilize multiple swarms
        swarms = [np.random.uniform(
            low=func.bounds.lb, 
            high=func.bounds.ub, 
            size=(population_size, self.dim)
        ) for _ in range(num_swarms)]
        
        fitness_swarms = [np.array([func(ind) for ind in swarm]) for swarm in swarms]
        self.budget -= population_size * num_swarms
        best_idx_swarms = [np.argmin(fitness) for fitness in fitness_swarms]
        best_solution_swarms = [swarms[i][best_idx_swarms[i]] for i in range(num_swarms)]

        # Evolution loop
        while self.budget > 0:
            for swarm_idx, (population, fitness, best_solution) in enumerate(zip(swarms, fitness_swarms, best_solution_swarms)):
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
                adaptive_mutation_prob = 0.1 + 0.4 * (fitness[best_idx_swarms[swarm_idx]] / fitness.mean())  # Adaptive mutation probability
                mutation_decay_factor = 1 - (self.budget / (self.budget + population_size))  # Dynamic mutation decay
                for i in range(population_size):
                    parent1, parent2, parent3 = parents[i], parents[(i + 1) % population_size], parents[(i + 2) % population_size]

                    # Crossover (blend crossover)
                    alpha = np.random.uniform(0, 1, self.dim)
                    centroid = (parent1 + parent2 + parent3) / 3
                    child = alpha * parent1 + (1 - alpha) * centroid

                    # Mutation (adaptive Gaussian mutation)
                    mutation_strength = mutation_decay_factor * 0.1 * (func.bounds.ub - func.bounds.lb) * (fitness[best_idx_swarms[swarm_idx]] / fitness.min())
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
                combined_population = np.vstack((population, offspring, [best_solution]))
                combined_fitness = np.hstack((fitness, offspring_fitness, [fitness[best_idx_swarms[swarm_idx]]]))
                best_individuals_idx = np.argsort(combined_fitness)[:population_size]
                swarms[swarm_idx] = combined_population[best_individuals_idx]
                fitness_swarms[swarm_idx] = combined_fitness[best_individuals_idx]
                
                # Update best solution for swarm
                best_idx_swarms[swarm_idx] = np.argmin(fitness_swarms[swarm_idx])
                best_solution_swarms[swarm_idx] = swarms[swarm_idx][best_idx_swarms[swarm_idx]]
        
        # Return the overall best found solution
        global_best_idx = np.argmin([func(best) for best in best_solution_swarms])
        return best_solution_swarms[global_best_idx]