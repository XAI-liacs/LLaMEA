import numpy as np

class QuantumInspiredOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        initial_population_size = 10
        population = np.random.uniform(lb, ub, (initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += initial_population_size

        while self.evaluations < self.budget:
            diversity = np.mean(np.std(population, axis=0))  # Calculate diversity
            neighborhood_size = (ub - lb) * (0.06 + 0.94 * (1 - self.evaluations / self.budget))  # Changed 0.04 to 0.06
            population_size = int(initial_population_size * (1 + diversity))  # Dynamic resizing based on diversity
            if population_size < len(population):
                population = population[:population_size]
                fitness = fitness[:population_size]
            else:
                new_individuals = np.random.uniform(lb, ub, (population_size - len(population), self.dim))
                new_fitness = np.array([func(ind) for ind in new_individuals])
                self.evaluations += len(new_individuals)
                population = np.vstack((population, new_individuals))
                fitness = np.concatenate((fitness, new_fitness))
            
            best_fitness = np.min(fitness)
            for i in range(population_size):
                if self.evaluations >= self.budget:
                    break
                partner_idx = np.random.choice(population_size)  # Select a random partner
                trial_vector = population[i] + 0.5 * (population[partner_idx] - population[i])  # DE-inspired crossover
                trial_vector = np.clip(trial_vector, lb, ub)
                new_fitness = func(trial_vector)
                self.evaluations += 1
                
                if new_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = new_fitness

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]