import numpy as np

class HybridGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        np.random.seed(0)
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        
        while evaluations < self.budget:
            # Selection
            selected_indices = np.random.choice(self.population_size, size=self.population_size, replace=True, p=self._selection_proba(fitness))
            selected_population = population[selected_indices]

            # Crossover
            offspring = []
            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected_population[i], selected_population[i+1]
                child1, child2 = self._crossover(parent1, parent2)
                offspring.append(child1)
                offspring.append(child2)

            # Mutation
            offspring = np.array(offspring)
            offspring = self._mutate(offspring)
            
            # Evaluate new offspring
            new_fitness = np.array([func(ind) for ind in offspring])
            evaluations += self.population_size
            
            # Elitism: keep the best solutions from each generation
            combined_population = np.vstack((population, offspring))
            combined_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population, fitness = combined_population[best_indices], combined_fitness[best_indices]
            
            # Re-evaluate elite individuals for robustness
            fitness[:self.population_size//2] = [func(population[i]) for i in range(self.population_size//2)]
        
        best_index = np.argmin(fitness)
        return population[best_index]

    def _selection_proba(self, fitness):
        # Using normalized inverse fitness proportional selection
        fitness = fitness - fitness.min() + 1e-9
        inverted_fitness = 1.0 / fitness
        return inverted_fitness / inverted_fitness.sum()

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand(self.dim)
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = alpha * parent2 + (1 - alpha) * parent1
            return child1, child2
        return parent1, parent2

    def _mutate(self, population):
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_rate:
                mutation_vector = np.random.randn(self.dim)
                scale_factor = np.random.rand() * 0.1
                population[i] += scale_factor * mutation_vector
                np.clip(population[i], self.bounds[0], self.bounds[1], out=population[i])
        return population