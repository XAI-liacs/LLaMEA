import numpy as np

class MGSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_prob = 0.8
        self.initial_mutation_rate = 0.1
        self.temperature = 1.0
        self.decay_rate = 0.99
        self.n_generations = budget // self.population_size

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_parents(self, population, fitness):
        probabilities = fitness / fitness.sum()
        indices = np.random.choice(range(self.population_size), size=2, p=probabilities)
        return population[indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            crossover_point = np.random.randint(1, self.dim - 1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def mutate(self, individual, lb, ub):
        mutation_strength = self.initial_mutation_rate * self.temperature
        mutation_vector = np.random.normal(scale=mutation_strength, size=self.dim)
        mutated = np.clip(individual + mutation_vector, lb, ub)
        return mutated

    def modular_search(self, population):
        for i in range(self.population_size):
            if np.random.rand() < 0.5:  # Encourage modular structures
                module_size = np.random.randint(1, self.dim // 2)
                module = np.random.uniform(-0.1, 0.1, module_size)
                for j in range(0, self.dim, module_size):
                    end_index = min(j + module_size, self.dim)
                    population[i][j:end_index] += module[:end_index - j]
        return population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_solution = None
        best_value = float('inf')

        for generation in range(self.n_generations):
            population = self.modular_search(population)
            fitness = self.evaluate_population(func, population)
            for i in range(0, self.population_size, 2):
                parent1, parent2 = self.select_parents(population, 1 / (fitness + 1e-9))
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1, lb, ub)
                child2 = self.mutate(child2, lb, ub)
                population[i], population[i + 1] = child1, child2

            generation_best = np.min(fitness)
            if generation_best < best_value:
                best_value = generation_best
                best_solution = population[np.argmin(fitness)]

            # Update temperature
            self.temperature *= self.decay_rate

        return best_solution