import numpy as np
from scipy.optimize import minimize

class HybridGeneticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.func_evals = 0

    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opp_pop = lb + ub - pop  # Quasi-oppositional initialization
        return np.vstack((pop, opp_pop))[:self.population_size]

    def evaluate_population(self, func, population):
        fitness = np.array([func(ind) for ind in population])
        self.func_evals += len(population)
        return fitness

    def select_parents(self, population, fitness):
        idx = np.random.choice(np.arange(self.population_size), size=self.population_size, replace=True, p=fitness / fitness.sum())
        return population[idx]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, child, bounds):
        if np.random.rand() < self.mutation_rate:
            idx = np.random.randint(self.dim)
            lb, ub = bounds.lb[idx], bounds.ub[idx]
            child[idx] = np.random.uniform(lb, ub)
        return child

    def local_search(self, func, individual, bounds):
        result = minimize(func, individual, bounds=list(zip(bounds.lb, bounds.ub)), method='L-BFGS-B', options={'maxiter': 50})
        return result.x if result.success else individual

    def enforce_periodicity(self, individual):
        period = 2
        individual[:period] = np.mean(individual[:period])
        individual[period:self.dim] = np.tile(individual[:period], self.dim // period)
        return individual

    def __call__(self, func):
        bounds = func.bounds
        population = self.initialize_population(bounds)
        fitness = self.evaluate_population(func, population)
        
        while self.func_evals < self.budget:
            parents = self.select_parents(population, fitness)
            offspring = []

            for i in range(0, len(parents), 2):
                parent1 = parents[i]
                parent2 = parents[i+1] if i+1 < len(parents) else parents[0]
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, bounds)
                child2 = self.mutate(child2, bounds)
                offspring.extend([child1, child2])

            offspring = np.array(offspring)
            offspring_fitness = self.evaluate_population(func, offspring)

            # Combine population and offspring, and select the best individuals
            combined_population = np.vstack((population, offspring))
            combined_fitness = np.concatenate((fitness, offspring_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

            # Local optimization on best individual to fine-tune using gradient information
            best_individual = population[0]
            if self.func_evals + 50 <= self.budget:
                population[0] = self.local_search(func, best_individual, bounds)
                fitness[0] = func(population[0])
                self.func_evals += 1

        return self.enforce_periodicity(population[0])