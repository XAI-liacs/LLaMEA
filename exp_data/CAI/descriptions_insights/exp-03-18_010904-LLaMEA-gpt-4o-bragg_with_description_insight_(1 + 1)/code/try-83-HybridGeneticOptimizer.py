import numpy as np
from scipy.optimize import minimize

class HybridGeneticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_rate = 0.12  # Slightly increased mutation rate to enhance diversity
        self.func_evals = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def evaluate_population(self, func, population):
        fitness = np.array([func(ind) for ind in population])
        self.func_evals += len(population)
        return fitness

    def select_parents(self, population, fitness):
        idx = np.random.choice(np.arange(self.population_size), size=self.population_size, replace=True, p=fitness / fitness.sum())
        return population[idx]

    def crossover(self, parent1, parent2):
        progress = self.func_evals / self.budget
        adaptive_crossover_rate = self.crossover_rate * (1 - progress) + 0.25 * progress + 0.1 * np.std(parent1 - parent2)  # Change 1: Modified adaptive crossover rate
        if np.random.rand() < adaptive_crossover_rate:
            point = np.random.randint(1, self.dim)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, child, bounds):
        progress = self.func_evals / self.budget
        self.mutation_rate = (0.15 + 0.05 * np.std(child)) * (1 - 0.5 * progress)  # Change 2: Adjusted mutation rate formula
        if np.random.rand() < self.mutation_rate:
            idx = np.random.randint(self.dim)
            lb, ub = bounds.lb[idx], bounds.ub[idx]
            child[idx] = np.random.uniform(lb, ub)
        return child

    def local_search(self, func, individual, bounds):
        result = minimize(func, individual, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B', options={'maxiter': 50})
        return result.x if result.success else individual

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(func, population)
        
        while self.func_evals < self.budget:
            best_individual = population[np.argmin(fitness)].copy()
            best_fitness = min(fitness)

            parents = self.select_parents(population, fitness)
            offspring = []

            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i+1] if i+1 < len(parents) else best_individual  # Use best_individual for elite recombination
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, bounds)
                child2 = self.mutate(child2, bounds)
                offspring.extend([child1, child2])

            offspring = np.array(offspring)
            offspring_fitness = self.evaluate_population(func, offspring)

            combined_population = np.vstack((population, offspring))
            combined_fitness = np.concatenate((fitness, offspring_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            population[-1] = best_individual
            fitness[-1] = best_fitness

            if self.func_evals + 50 <= self.budget:
                population[0] = self.local_search(func, population[0], bounds)
                fitness[0] = func(population[0])
                self.func_evals += 1

        return population[0]