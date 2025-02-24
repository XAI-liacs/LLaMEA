import numpy as np
from scipy.optimize import minimize

class RHPGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.crossover_prob = 0.8
        self.mutation_rate = 0.2
        self.local_search_prob = 0.15
        self.n_generations = budget // self.population_size

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.population_size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_parents(self, population, fitness):
        probabilities = fitness / fitness.sum()
        indices = np.random.choice(range(self.population_size), size=2, p=probabilities)
        return population[indices]

    def crossover(self, parent1, parent2, generation, max_gen):
        dynamic_crossover_prob = self.crossover_prob * ((max_gen - generation) / max_gen) + 0.1
        if np.random.rand() < dynamic_crossover_prob:
            crossover_point = np.random.randint(1, self.dim - 1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def adaptive_mutate(self, individual, lb, ub, gen, max_gen):
        mutation_strength = self.mutation_rate * ((1 - gen / max_gen) ** 2) * 0.95
        mutation_vector = np.random.normal(scale=mutation_strength, size=self.dim)
        mutated = np.clip(individual + mutation_vector, lb, ub)
        return mutated

    def local_search(self, func, individual, lb, ub):
        def bounded_func(x):
            return func(np.clip(x, lb, ub))
        result = minimize(bounded_func, individual, method='L-BFGS-B')
        return result.x

    def improved_periodicity_heuristic(self, population, gen, max_gen):
        for i in range(self.population_size):
            if np.random.rand() < 0.6:
                adaptive_period = np.random.randint(1, max(2, self.dim // (2 + int(3 * (gen / max_gen)))))
                pattern = population[i][:adaptive_period]
                for j in range(adaptive_period, self.dim, adaptive_period):
                    end_index = min(j + adaptive_period, self.dim)
                    population[i][j:end_index] = pattern[:end_index - j]
        return population

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self.initialize_population(lb, ub)
        best_solution = None
        best_value = float('inf')

        for generation in range(self.n_generations):
            population = self.improved_periodicity_heuristic(population, generation, self.n_generations)
            fitness = self.evaluate_population(func, population)
            for i in range(0, self.population_size, 2):
                parent1, parent2 = self.select_parents(population, 1 / (fitness + 1e-9))
                child1, child2 = self.crossover(parent1, parent2, generation, self.n_generations)
                child1 = self.adaptive_mutate(child1, lb, ub, generation, self.n_generations)
                child2 = self.adaptive_mutate(child2, lb, ub, generation, self.n_generations)
                population[i], population[i+1] = child1, child2

            dynamic_local_search_prob = self.local_search_prob * (generation / self.n_generations)
            if np.random.rand() < dynamic_local_search_prob:
                for i in range(self.population_size):
                    population[i] = self.local_search(func, population[i], lb, ub)

            generation_best = np.min(fitness)
            if generation_best < best_value:
                best_value = generation_best
                best_solution = population[np.argmin(fitness)]

        return best_solution