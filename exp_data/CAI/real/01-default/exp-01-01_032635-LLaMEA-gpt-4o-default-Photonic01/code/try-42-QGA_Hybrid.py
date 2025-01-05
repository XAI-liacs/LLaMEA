import numpy as np

class QGA_Hybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 2)
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_value = np.inf
        self.bounds = None
        self.crossover_prob = 0.8
        self.mutation_prob = 0.1

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_initialization(self):
        # Apply quantum random walk for diverse initialization
        qwalk = np.random.normal(0, 1, (self.population_size, self.dim))
        self.population = self.population + qwalk * 0.05
        lb, ub = self.bounds
        self.population = np.clip(self.population, lb, ub)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            value = func(self.population[i])
            if value < self.fitness[i]:
                self.fitness[i] = value
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = self.population[i].copy()

    def select_parents(self):
        # Tournament selection
        idx = np.random.choice(np.arange(self.population_size), size=(self.population_size, 2), replace=True)
        better_idx = np.argmin(self.fitness[idx], axis=1)
        parents = self.population[idx[np.arange(self.population_size), better_idx]]
        return parents

    def crossover(self, parents):
        # Uniform crossover
        children = np.empty_like(parents)
        for i in range(0, self.population_size, 2):
            if np.random.rand() < self.crossover_prob:
                mask = np.random.rand(self.dim) > 0.5
                children[i] = np.where(mask, parents[i], parents[i+1])
                children[i+1] = np.where(mask, parents[i+1], parents[i])
            else:
                children[i], children[i+1] = parents[i], parents[i+1]
        return children

    def mutate(self, children):
        # Gaussian mutation
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_prob:
                noise = np.random.normal(0, 0.1, self.dim)
                children[i] += noise
                lb, ub = self.bounds
                children[i] = np.clip(children[i], lb, ub)
        return children

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        self.quantum_initialization()
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            parents = self.select_parents()
            children = self.crossover(parents)
            children = self.mutate(children)
            self.population = children

        return self.best_solution, self.best_value