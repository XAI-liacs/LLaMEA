import numpy as np

class AdaptiveQuantumGeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, min(100, budget // 5))
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.mutation_step = 0.05
        self.best_individual = None
        self.best_fitness = float('inf')
        self.interference_prob = 0.1

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self, func):
        fitness = np.array([func(individual) for individual in self.population])
        for i, f in enumerate(fitness):
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_individual = self.population[i]
        return fitness

    def select_parents(self, fitness):
        probabilities = fitness / fitness.sum()
        indices = np.random.choice(self.population_size, size=2, p=probabilities)
        return self.population[indices[0]], self.population[indices[1]]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand(self.dim)
            child = alpha * parent1 + (1 - alpha) * parent2
        else:
            child = parent1 if np.random.rand() < 0.5 else parent2
        return child

    def mutate(self, individual, lb, ub):
        if np.random.rand() < self.mutation_rate:
            quantum_step = (np.random.rand(self.dim) - 0.5) * 2 * self.mutation_step
            individual += quantum_step * (self.best_individual - individual)
            individual = np.clip(individual, lb, ub)
        return individual

    def apply_social_learning(self, lb, ub):
        for i in range(self.population_size):
            if np.random.rand() < self.interference_prob:
                random_vector = lb + (ub - lb) * np.random.rand(self.dim)
                self.population[i] = np.mean([self.population[i], random_vector], axis=0)
                self.population[i] = np.clip(self.population[i], lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate_population(func)
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(fitness)
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                new_population.append(self.mutate(child1, lb, ub))
                new_population.append(self.mutate(child2, lb, ub))

            self.population = np.array(new_population)
            self.apply_social_learning(lb, ub)

        return self.best_individual, self.best_fitness