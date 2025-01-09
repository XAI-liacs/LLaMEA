import numpy as np

class HybridGeneticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.local_search_rate = 0.2
        self.lb = -5.0
        self.ub = 5.0
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            # Evaluate fitness of the population
            for i in range(self.population_size):
                if self.fitness[i] == np.inf:
                    self.fitness[i] = func(self.population[i])
                    evaluations += 1
                    if evaluations >= self.budget:
                        break
            
            # Selection: Tournament selection
            parents = self.tournament_selection()

            # Crossover: Simulated Binary Crossover
            offspring = self.crossover(parents)
            
            # Mutation: Gaussian mutation
            self.mutation(offspring)
            
            # Local Search
            self.local_search(offspring, func, evaluations)
            
            # Replace worst solutions with new offspring
            self.replace(offspring, func)
        
        # Return the best solution found
        best_index = np.argmin(self.fitness)
        return self.population[best_index], self.fitness[best_index]
    
    def tournament_selection(self):
        parents = []
        for _ in range(self.population_size):
            i, j = np.random.choice(self.population_size, 2, replace=False)
            if self.fitness[i] < self.fitness[j]:
                parents.append(self.population[i])
            else:
                parents.append(self.population[j])
        return np.array(parents)
    
    def crossover(self, parents):
        offspring = []
        for i in range(0, self.population_size, 2):
            if np.random.rand() < self.crossover_rate:
                parent1, parent2 = parents[i], parents[i+1]
                beta = np.random.rand(self.dim) * 2 - 1
                child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
                child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
                child1 = np.clip(child1, self.lb, self.ub)
                child2 = np.clip(child2, self.lb, self.ub)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parents[i], parents[i+1]])
        return offspring

    def mutation(self, offspring):
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation = np.random.normal(0, 0.1, self.dim)
                offspring[i] += mutation
                offspring[i] = np.clip(offspring[i], self.lb, self.ub)
    
    def local_search(self, offspring, func, evaluations):
        for i in range(self.population_size):
            if np.random.rand() < self.local_search_rate:
                for _ in range(5):  # perform a few local search steps
                    step = np.random.normal(0, 0.1, self.dim)
                    candidate = offspring[i] + step
                    candidate = np.clip(candidate, self.lb, self.ub)
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    if candidate_fitness < func(offspring[i]):
                        offspring[i] = candidate
                        if evaluations >= self.budget:
                            return
    
    def replace(self, offspring, func):
        for i in range(self.population_size):
            if func(offspring[i]) < self.fitness[i]:
                self.population[i] = offspring[i]
                self.fitness[i] = func(offspring[i])