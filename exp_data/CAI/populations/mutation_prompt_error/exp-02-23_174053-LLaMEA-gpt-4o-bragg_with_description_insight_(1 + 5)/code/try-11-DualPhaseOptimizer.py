import numpy as np
from scipy.optimize import minimize

class DualPhaseOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            # Selection
            selected_indices = np.random.choice(self.population_size, self.population_size // 2, replace=False)
            parents = population[selected_indices]

            # Crossover
            offspring = np.empty((0, self.dim))
            for i in range(0, len(parents), 2):
                parent1, parent2 = parents[i], parents[i + 1]
                mask = np.random.rand(self.dim) < self.crossover_rate
                child1 = np.where(mask, parent1, parent2)
                child2 = np.where(mask, parent2, parent1)
                offspring = np.vstack((offspring, child1, child2))

            # Mutation
            mutation_mask = np.random.rand(*offspring.shape) < self.mutation_rate
            mutation_values = np.random.uniform(bounds[0], bounds[1], offspring.shape)
            offspring = np.where(mutation_mask, mutation_values, offspring)

            # Calculate fitness for the new offspring
            new_fitness = np.array([func(ind) for ind in offspring])
            eval_count += len(offspring)

            # Replacement
            population = np.vstack((population, offspring))
            fitness = np.append(fitness, new_fitness)
            # Select top individuals
            best_indices = np.argsort(fitness)[-self.population_size:]
            population = population[best_indices]
            fitness = fitness[best_indices]

        # Local optimization on the best found solution
        best_index = np.argmax(fitness)
        best_solution = population[best_index]

        # Custom periodic-aware local search for fine-tuning
        best_solution = self.periodic_local_search(func, best_solution, bounds)

        return best_solution

    def periodic_local_search(self, func, solution, bounds):
        period = self.dim // 2
        perturbation_scale = 0.1
        perturbation = np.zeros(self.dim)
        
        for _ in range(10):  # Limit the local search to 10 iterations
            for i in range(0, self.dim, period):
                segment = solution[i:i + period]
                mean_value = np.mean(segment)
                perturbation[i:i + period] = perturbation_scale * mean_value * np.sin(np.arange(period))
            
            candidate_solution = solution + perturbation
            candidate_solution = np.clip(candidate_solution, bounds[0], bounds[1])
            if func(candidate_solution) > func(solution):
                solution = candidate_solution
        
        return solution