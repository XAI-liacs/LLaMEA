import numpy as np

class HybridGADiffEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 20
        self.crossover_rate = 0.7
        self.mutation_factor = 0.8
        self.elitism_ratio = 0.1
        self.evaluations = 0
    
    def __call__(self, func):
        population = self._initialize_population()
        fitness = self._evaluate_population(func, population)
        
        while self.evaluations < self.budget:
            new_population = self._elitism_selection(population, fitness)
            for _ in range(self.population_size - len(new_population)):
                parents = self._select_parents(fitness)
                offspring = self._crossover(parents)
                offspring = self._mutate(offspring)
                new_population.append(offspring)
            
            population = np.array(new_population)
            fitness = self._evaluate_population(func, population)
        
        best_index = np.argmin(fitness)
        return population[best_index]
    
    def _initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
    
    def _evaluate_population(self, func, population):
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            fitness[i] = func(population[i])
            self.evaluations += 1
            if self.evaluations >= self.budget:
                break
        return fitness
    
    def _elitism_selection(self, population, fitness):
        num_elites = int(self.elitism_ratio * self.population_size)
        elite_indices = fitness.argsort()[:num_elites]
        return [population[i] for i in elite_indices]
    
    def _select_parents(self, fitness):
        idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
        if fitness[idx1] < fitness[idx2]:
            return idx1, idx2
        return idx2, idx1
    
    def _crossover(self, parents):
        parent1, parent2 = parents
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(0, self.dim)
            offspring = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        else:
            offspring = parent1
        return offspring
    
    def _mutate(self, individual):
        mutation = np.random.randn(self.dim) * self.mutation_factor
        mutant = np.clip(individual + mutation, self.lb, self.ub)
        return mutant