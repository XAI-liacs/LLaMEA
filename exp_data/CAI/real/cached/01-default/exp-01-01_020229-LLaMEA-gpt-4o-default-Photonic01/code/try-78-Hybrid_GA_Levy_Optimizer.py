import numpy as np

class Hybrid_GA_Levy_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.crossover_rate = 0.8
        self.mutation_rate = 0.2
        self.alpha = 0.01  # Step size for Levy flight
        self.beta = 1.5  # Parameter for Levy distribution
        
    def levy_flight(self):
        u = np.random.normal(0, 1, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.power(np.abs(v), 1/self.beta)
        return self.alpha * step
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            # Selection via tournament
            new_population = []
            for _ in range(self.population_size):
                candidates = np.random.choice(self.population_size, 3, replace=False)
                best_candidate = candidates[np.argmin(fitness[candidates])]
                new_population.append(population[best_candidate])
            population = np.array(new_population)

            # Crossover
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.crossover_rate:
                    crossover_point = np.random.randint(1, self.dim)
                    population[i, crossover_point:], population[i+1, crossover_point:] = \
                    population[i+1, crossover_point:], population[i, crossover_point:]

            # Mutation with Levy flight
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    population[i] += self.levy_flight()
                    population[i] = np.clip(population[i], lb, ub)

            # Evaluate new population
            population_fitness = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            # Elitism: keep the best individuals
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((population_fitness, fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            population = combined_population[best_indices]
            fitness = combined_fitness[best_indices]

        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]