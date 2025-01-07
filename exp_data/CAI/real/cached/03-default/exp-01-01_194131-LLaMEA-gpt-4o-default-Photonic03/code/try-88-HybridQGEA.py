import numpy as np

class HybridQGEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.population = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_individual = None
        self.best_fitness = float('inf')
        self.crossover_prob = 0.8
        self.mutation_scale = 0.02
        self.evaluations = 0

    def quantum_mutation(self, individual):
        delta = np.random.normal(0, 1, self.dim) * self.mutation_scale
        mutated_individual = individual + delta
        return mutated_individual

    def crossover(self, parent1, parent2):
        mask = np.random.rand(self.dim) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def select_parents(self):
        idx1, idx2 = np.random.choice(self.population_size, size=2, replace=False)
        return self.population[idx1], self.population[idx2]

    def _adapt_mutation_scale(self):
        if self.evaluations % (self.budget // 5) == 0:
            self.mutation_scale = max(0.01, self.mutation_scale * 0.9)

    def __call__(self, func):
        bounds_low, bounds_high = func.bounds.lb, func.bounds.ub
        self.population = bounds_low + (bounds_high - bounds_low) * np.random.rand(self.population_size, self.dim)
        
        for i in range(self.population_size):
            score = func(self.population[i])
            self.fitness[i] = score
            if score < self.best_fitness:
                self.best_individual = self.population[i]
                self.best_fitness = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_individual

        while self.evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents()
                if np.random.rand() < self.crossover_prob:
                    child1 = self.crossover(parent1, parent2)
                    child2 = self.crossover(parent2, parent1)
                else:
                    child1, child2 = parent1, parent2
                
                child1 = np.clip(child1, bounds_low, bounds_high)
                child2 = np.clip(child2, bounds_low, bounds_high)

                if np.random.rand() < 0.5:
                    child1 = self.quantum_mutation(child1)
                    child2 = self.quantum_mutation(child2)
                
                new_population.append(child1)
                new_population.append(child2)

            self.population = np.array(new_population)
            for i in range(self.population_size):
                score = func(self.population[i])
                if score < self.fitness[i]:
                    self.fitness[i] = score
                if score < self.best_fitness:
                    self.best_individual = self.population[i]
                    self.best_fitness = score
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    break
            self._adapt_mutation_scale()

        return self.best_individual