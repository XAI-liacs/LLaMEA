import numpy as np

class BiLevelAdaptiveGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_rate = 0.1
        self.elitism_rate = 0.1
        self.evaluations = 0

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = self.initialize_population(bounds)
        scores = np.array([self.evaluate(func, ind) for ind in population])
        
        while self.evaluations < self.budget:
            # Selection
            selected_indices = self.roulette_wheel_selection(scores)
            offspring = population[selected_indices]
            
            # Crossover
            offspring = self.crossover(offspring, bounds)
            
            # Mutation
            offspring = self.mutate(offspring, scores[selected_indices], bounds)
            
            # Evaluate offspring
            offspring_scores = np.array([self.evaluate(func, ind) for ind in offspring])
            
            # Elitism
            top_indices = np.argsort(scores)[:int(self.population_size * self.elitism_rate)]
            combined_population = np.concatenate((population[top_indices], offspring))
            combined_scores = np.concatenate((scores[top_indices], offspring_scores))
            
            # Survival selection
            best_indices = np.argsort(combined_scores)[:self.population_size]
            population = combined_population[best_indices]
            scores = combined_scores[best_indices]
        
        best_index = np.argmin(scores)
        return population[best_index]

    def initialize_population(self, bounds):
        return np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    def evaluate(self, func, individual):
        if self.evaluations >= self.budget:
            return float('inf')
        self.evaluations += 1
        return func(individual)

    def roulette_wheel_selection(self, scores):
        fitness = 1 / (1 + scores)
        probabilities = fitness / np.sum(fitness)
        return np.random.choice(self.population_size, self.population_size, p=probabilities)

    def crossover(self, population, bounds):
        offspring = population.copy()
        np.random.shuffle(offspring)
        for i in range(0, self.population_size - 1, 2):
            if np.random.rand() < 0.9:  # Crossover probability
                alpha = np.random.rand(self.dim)
                offspring[i] = alpha * population[i] + (1 - alpha) * population[i + 1]
                offspring[i + 1] = alpha * population[i + 1] + (1 - alpha) * population[i]
        return np.clip(offspring, bounds[:, 0], bounds[:, 1])

    def mutate(self, population, fitness_ranks, bounds):
        mutation_strength = self.mutation_rate * (1.0 + 0.5 * (fitness_ranks / np.max(fitness_ranks)))
        for i in range(self.population_size):
            if np.random.rand() < mutation_strength[i]:
                disturbance = np.random.randn(self.dim) * (bounds[:, 1] - bounds[:, 0]) * 0.1
                population[i] = np.clip(population[i] + disturbance, bounds[:, 0], bounds[:, 1])
        return population