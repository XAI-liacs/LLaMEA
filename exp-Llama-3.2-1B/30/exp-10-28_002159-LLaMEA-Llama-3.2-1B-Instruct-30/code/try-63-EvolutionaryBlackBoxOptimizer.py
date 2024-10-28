import random
import numpy as np

class EvolutionaryBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.population = self.init_population()
        self.fitness_scores = np.zeros((self.population_size, self.dim))
        self.search_spaces = [(-5.0, 5.0)] * self.dim

    def init_population(self):
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(self.search_spaces[i][0], self.search_spaces[i][1]) for i in range(self.dim)]
            population.append(individual)
        return population

    def __call__(self, func):
        def fitness(individual):
            return func(individual)

        def fitness_scores(individual):
            return np.array([fitness(individual) for _ in range(self.population_size)])

        def mutate(individual):
            if random.random() < 0.3:
                index = random.randint(0, self.dim - 1)
                self.search_spaces[index] = (self.search_spaces[index][0] + random.uniform(-1, 1), self.search_spaces[index][1] + random.uniform(-1, 1))
            return individual

        for _ in range(self.budget):
            for i, individual in enumerate(self.population):
                fitness_scores[i] = fitness_scores[i]
                fitness(individual)
                best_individual = self.population[np.argmax(fitness_scores)]
                new_individual = mutate(individual)
                self.population[i] = new_individual
                self.fitness_scores[i] = fitness_scores[i]

        return self.population

    def evaluate(self, func):
        return func(np.array(self.population))

# Code: 